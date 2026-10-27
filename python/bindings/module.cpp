/**
 * @file module.cpp
 * @brief Nexus kamu API'sinin pybind11 kabugu. Torch isimleri yok.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "image_write.hpp"
#include "matrix_pro/cpu/gemm.hpp"
#include "matrix_pro/nn/conv.hpp"
#include "nexus_optim/core/thread_pool.hpp"
#include "nexus/custom/loss.hpp"
#include "nexus/custom/metric.hpp"
#include "nexus/custom/microtape.hpp"
#include "nexus/custom/tape.hpp"
#include "nexus/custom/optimizer.hpp"
#include "nexus/data/datasets.hpp"
#include "nexus/loss/losses.hpp"
#include "nexus/metrics/metrics.hpp"
#include "nexus/model/nxs_format.hpp"
#include "nexus/nn/activations.hpp"
#include "nexus/nn/modules.hpp"
#include "nexus/optim/optimizers.hpp"
#include "nexus/optim/schedulers.hpp"
#include "nexus/rl/actor_critic.hpp"
#include "nexus/rl/dqn.hpp"
#include "nexus/rl/envs.hpp"
#include "nexus/rl/league.hpp"
#include "nexus/train/language.hpp"
#include "nexus/rl/multi_agent_trainer.hpp"
#include "nexus/rl/ppo.hpp"
#include "nexus/rl/prioritized_replay.hpp"
#include "nexus/rl/replay_buffer.hpp"
#include "nexus/rl/rollout_buffer.hpp"
#include "nexus/rl/sac.hpp"
#include "nexus/rl/vector_env.hpp"
#include "nexus/train/distributed.hpp"
#include "nexus/train/trainer.hpp"
#include "nexusdata/image/image.hpp"
#include "nexusdata/simd/image_ops.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace {

std::vector<float> host_floats(const nexus::Tensor& tensor) {
    const float* data = tensor.data();
    return std::vector<float>(data, data + tensor.numel());
}

std::size_t flat_index(const nexus_model::Tensor& tensor, const std::vector<std::size_t>& index) {
    if (index.empty() || index.size() > 4 || index.size() != tensor.rank()) {
        throw std::invalid_argument("Tensor index rank");
    }
    std::size_t flat = 0;
    std::size_t stride = 1;
    for (std::size_t axis = index.size(); axis > 0; --axis) {
        const std::size_t coord = index[axis - 1];
        if (coord >= tensor.size(axis - 1)) {
            throw std::out_of_range("Tensor index");
        }
        flat += coord * stride;
        stride *= tensor.size(axis - 1);
    }
    return flat;
}

py::list parameter_views(nexus::nn::Module& module) {
    py::list out;
    const auto parameters = module.parameters();
    for (std::size_t index = 0; index < parameters.size(); ++index) {
        py::dict item;
        item["name"] = std::to_string(index);
        item["shape"] = parameters[index]->data.shape_vec();
        item["values"] = parameters[index]->data;
        item["grad"] = parameters[index]->grad;
        out.append(std::move(item));
    }
    return out;
}

py::dict step_dict(const nexus::rl::StepResult& result) {
    py::dict out;
    out["observation"] = host_floats(result.observation);
    out["reward"] = result.reward;
    out["terminated"] = result.terminated;
    out["truncated"] = result.truncated;
    out["info"] = result.info;
    return out;
}

void attach_python_payload() {
    py::module_ fns = py::module_::import("nexus.fnstore");
    const std::string raw = fns.attr("pack")().cast<std::string>();
    nexus::nxs::set_python_payload(std::vector<std::uint8_t>(raw.begin(), raw.end()));
}

void restore_python_payload(const std::string& path) {
    const std::vector<std::uint8_t> bytes = nexus::nxs::load_archive(path).python_bytes;
    if (bytes.empty()) {
        return;
    }
    py::bytes blob(reinterpret_cast<const char*>(bytes.data()), static_cast<py::ssize_t>(bytes.size()));
    py::module_::import("nexus.fnstore").attr("restore")(blob);
}

class PyEnvironment final : public nexus::rl::Environment {
 public:
    PyEnvironment(int observation_dim, int actions, bool discrete)
        : observation_(nexus::rl::Space::box(observation_dim, -1.0e6f, 1.0e6f)),
          action_(discrete ? nexus::rl::Space::discrete(actions) : nexus::rl::Space::box(std::max(actions, 1), -1.0f, 1.0f)),
          discrete_(discrete) {
        if (observation_dim <= 0 || (discrete && actions <= 1)) {
            throw std::invalid_argument("PyEnvironment dimensions are invalid");
        }
    }

    [[nodiscard]] const nexus::rl::Space& observation_space() const override { return observation_; }
    [[nodiscard]] const nexus::rl::Space& action_space() const override { return action_; }

    nexus::Tensor reset(std::uint32_t seed) override {
        py::gil_scoped_acquire gil;
        const py::object method = py::get_override(static_cast<nexus::rl::Environment*>(this), "reset");
        if (!method) {
            throw std::logic_error("python govdesi bu surecte yok");
        }
        const auto values = method(seed).cast<std::vector<float>>();
        return nexus::Tensor(nexus::Shape{values.size()}, std::span<const float>(values.data(), values.size()));
    }

    nexus::rl::StepResult step(const nexus::Tensor& action) override {
        py::gil_scoped_acquire gil;
        const py::object method = py::get_override(static_cast<nexus::rl::Environment*>(this), "step");
        if (!method) {
            throw std::logic_error("python govdesi bu surecte yok");
        }
        py::object py_action;
        if (discrete_) {
            py_action = py::int_(static_cast<int>(std::lround(action[0])));
        } else {
            py_action = py::cast(std::vector<float>(action.data(), action.data() + action.numel()));
        }
        const py::dict result = method(py_action);
        const auto values = result["observation"].cast<std::vector<float>>();
        nexus::rl::StepResult out;
        out.observation = nexus::Tensor(nexus::Shape{values.size()}, std::span<const float>(values.data(), values.size()));
        out.reward = result["reward"].cast<float>();
        out.terminated = result["terminated"].cast<bool>();
        out.truncated = result.contains("truncated") ? result["truncated"].cast<bool>() : false;
        out.info = result.contains("info") ? result["info"].cast<std::string>() : std::string();
        return out;
    }

 private:
    nexus::rl::Space observation_;
    nexus::rl::Space action_;
    bool discrete_ = true;
};

class PyCallback final : public nexus::callbacks::Callback {
 public:
    void on_epoch_end(const nexus::callbacks::EpochState& state) override {
        py::gil_scoped_acquire gil;
        if (const py::function method = py::get_override(static_cast<nexus::callbacks::Callback*>(this), "on_epoch_end")) {
            py::dict view;
            view["epoch"] = state.epoch;
            view["train_loss"] = state.train_loss;
            view["val_loss"] = state.val_loss;
            view["accuracy"] = state.accuracy;
            view["f1"] = state.f1;
            method(view);
        }
    }

    [[nodiscard]] bool stop() const override {
        py::gil_scoped_acquire gil;
        if (const py::function method = py::get_override(const_cast<nexus::callbacks::Callback*>(static_cast<const nexus::callbacks::Callback*>(this)), "stop")) {
            return method().cast<bool>();
        }
        return false;
    }
};

class PythonLayer final : public nexus::nn::Module {
 public:
    PythonLayer(std::string name, std::vector<std::vector<std::size_t>> shapes, py::function forward, py::object backward)
        : name_(std::move(name)), forward_(std::move(forward)), backward_(std::move(backward)) {
        if (name_.empty()) {
            throw std::invalid_argument("PythonModule name is empty");
        }
        for (const std::vector<std::size_t>& shape : shapes) {
            if (shape.empty()) {
                throw std::invalid_argument("PythonModule parameter shape is empty");
            }
            owned_.push_back(std::make_unique<nexus_model::Parameter>(nexus_model::Tensor::zeros(shape)));
        }
    }

    nexus_model::Tensor forward(const nexus_model::Tensor& input) override {
        py::gil_scoped_acquire gil;
        py::list params;
        for (const std::unique_ptr<nexus_model::Parameter>& parameter : owned_) {
            params.append(parameter->data);
        }
        return forward_(input, params).cast<nexus_model::Tensor>();
    }

    nexus_model::Tensor backward(const nexus_model::Tensor& grad_output) override {
        py::gil_scoped_acquire gil;
        if (backward_.is_none()) {
            throw std::logic_error("python govdesi bu surecte yok");
        }
        py::list params;
        for (const std::unique_ptr<nexus_model::Parameter>& parameter : owned_) {
            py::dict item;
            item["values"] = parameter->data;
            item["grad"] = parameter->grad;
            params.append(std::move(item));
        }
        return py::cast<py::function>(backward_)(grad_output, params).cast<nexus_model::Tensor>();
    }

    [[nodiscard]] std::vector<nexus_model::Parameter*> parameters() const override {
        std::vector<nexus_model::Parameter*> out;
        out.reserve(owned_.size());
        for (const std::unique_ptr<nexus_model::Parameter>& parameter : owned_) {
            out.push_back(parameter.get());
        }
        return out;
    }

    [[nodiscard]] nexus::nxs::LayerSpec nxs_spec() const override {
        return nexus::nxs::spec_of(nexus::nxs::LayerKind::PythonModule, name_);
    }

 private:
    std::string name_;
    py::function forward_;
    py::object backward_;
    std::vector<std::unique_ptr<nexus_model::Parameter>> owned_;
};

template <typename Env>
void bind_env(py::module_& rl, const char* name) {
    py::class_<Env, nexus::rl::Environment, std::shared_ptr<Env>>(rl, name)
        .def(py::init<>())
        .def("observation_dim", &Env::observation_dim)
        .def("action_count", &Env::action_count)
        .def("reset", [](Env& env, std::uint32_t seed) { return host_floats(env.reset(seed)); })
        .def("step", [](Env& env, int action) { return step_dict(env.step(nexus::rl::discrete_action_tensor(action))); });
}

}  // namespace

void bind_faz17(py::module_& m);  // interop.cpp

PYBIND11_MODULE(_core, m) {
    bind_faz17(m);
    nexus_optim::ThreadPool::use_caller_thread();
    m.doc() = "Nexus Python kabugu. Cagri adlari C++ yuzeyinin kendisidir.";
    m.attr("__version__") = "1.0.0";

    py::class_<nexus_model::Tensor>(m, "Tensor", py::buffer_protocol())
        .def(py::init([](std::vector<std::size_t> shape) { return nexus_model::Tensor::zeros(shape); }))
        .def("fill", &nexus_model::Tensor::fill)
        .def("numel", &nexus_model::Tensor::numel)
        .def_property_readonly("shape", [](const nexus_model::Tensor& tensor) { return tensor.shape_vec(); })
        .def("reshape", [](const nexus_model::Tensor& tensor, std::vector<std::size_t> shape) {
            if (shape.empty() || shape.size() > 8) {
                throw std::invalid_argument("reshape rank");
            }
            std::size_t known = 1;
            int infer = -1;
            for (std::size_t axis = 0; axis < shape.size(); ++axis) {
                if (shape[axis] == static_cast<std::size_t>(-1)) {
                    if (infer >= 0) {
                        throw std::invalid_argument("reshape has two inferred dimensions");
                    }
                    infer = static_cast<int>(axis);
                } else {
                    known *= shape[axis];
                }
            }
            if (infer >= 0) {
                if (known == 0 || tensor.numel() % known != 0) {
                    throw std::invalid_argument("reshape cannot infer dimension");
                }
                shape[static_cast<std::size_t>(infer)] = tensor.numel() / known;
            }
            return tensor.view_as(shape.data(), static_cast<std::uint8_t>(shape.size()));
        })
        .def("__getitem__", [](const nexus_model::Tensor& tensor, py::object key) {
            if (py::isinstance<py::tuple>(key) || py::isinstance<py::list>(key)) {
                return tensor[flat_index(tensor, key.cast<std::vector<std::size_t>>())];
            }
            return tensor[key.cast<std::size_t>()];
        })
        .def("__setitem__", [](nexus_model::Tensor& tensor, py::object key, float value) {
            if (py::isinstance<py::tuple>(key) || py::isinstance<py::list>(key)) {
                tensor[flat_index(tensor, key.cast<std::vector<std::size_t>>())] = value;
                return;
            }
            tensor[key.cast<std::size_t>()] = value;
        })
        .def("copy_from", [](nexus_model::Tensor& tensor, const std::vector<float>& values) {
            if (values.size() != tensor.numel()) {
                throw std::invalid_argument("Tensor copy_from length mismatch");
            }
            if (!values.empty()) {
                std::memcpy(tensor.data(), values.data(), values.size() * sizeof(float));
            }
        })
        .def("to_list", [](const nexus_model::Tensor& tensor) {
            return std::vector<float>(tensor.data(), tensor.data() + tensor.numel());
        })
        .def_buffer([](nexus_model::Tensor& tensor) {
            const std::vector<std::size_t> shape = tensor.shape_vec();
            std::vector<py::ssize_t> py_shape(shape.begin(), shape.end());
            std::vector<py::ssize_t> strides(shape.size());
            py::ssize_t stride = static_cast<py::ssize_t>(sizeof(float));
            for (std::size_t i = shape.size(); i > 0; --i) {
                strides[i - 1] = stride;
                stride *= static_cast<py::ssize_t>(shape[i - 1]);
            }
            return py::buffer_info(tensor.data(), sizeof(float), py::format_descriptor<float>::format(),
                                   static_cast<py::ssize_t>(shape.size()), py_shape, strides);
        });

    m.def("load", [](const std::string& path) {
        restore_python_payload(path);
        return nexus::nn::Sequential::load_nxs(path);
    });
    m.def("save", [](const nexus::nn::Sequential& model, const std::string& path) {
        attach_python_payload();
        model.save_nxs(path);
    });

    auto nn = m.def_submodule("nn");
    py::class_<nexus::nn::Module, std::shared_ptr<nexus::nn::Module>>(nn, "Module")
        .def("zero_grad", &nexus::nn::Module::zero_grad)
        .def("parameters", [](nexus::nn::Module& self) { return parameter_views(self); });

#define NEXUS_LAYER(Type, ...)                                                                                        \
    py::class_<nexus::nn::Type, nexus::nn::Module, std::shared_ptr<nexus::nn::Type>>(nn, #Type)                       \
        .def(py::init<__VA_ARGS__>())                                                                                 \
        .def("forward", &nexus::nn::Type::forward)                                                                    \
        .def("backward", &nexus::nn::Type::backward)                                                                  \
        .def("train", &nexus::nn::Type::train)                                                                        \
        .def("eval", &nexus::nn::Type::eval)

    NEXUS_LAYER(Linear, std::size_t, std::size_t, bool);
    NEXUS_LAYER(Conv2d, int, int, int, int, int, bool);
    NEXUS_LAYER(MultiHeadAttention, int, int);
    NEXUS_LAYER(LSTM, int, int);
    NEXUS_LAYER(GRU, int, int);
    NEXUS_LAYER(TransformerEncoder, int, int, int);
    NEXUS_LAYER(Embedding, int, int, int);
    NEXUS_LAYER(BatchNorm1d, int);
    NEXUS_LAYER(LayerNorm, std::size_t, float);
    NEXUS_LAYER(ReLU);
    NEXUS_LAYER(Dropout, float);
    NEXUS_LAYER(MaxPool2d, int, int, int);
    NEXUS_LAYER(AvgPool2d, int, int, int);
    NEXUS_LAYER(GlobalAvgPool2d);
    NEXUS_LAYER(Flatten);
    NEXUS_LAYER(BatchNorm2d, int);
    NEXUS_LAYER(Conv1d, int, int, int, int, int, bool);
    NEXUS_LAYER(Conv3d, int, int, int, int, int, bool);
    NEXUS_LAYER(DepthwiseConv2d, int, int, int, int);
    NEXUS_LAYER(GroupNorm, int, int);
    NEXUS_LAYER(InstanceNorm, int);
    NEXUS_LAYER(RNN, int, int);
    NEXUS_LAYER(PositionalEncoding, int, int, bool);
    NEXUS_LAYER(CausalTransformerBlock, int, int, int);
#undef NEXUS_LAYER

    py::class_<nexus::nn::TransformerDecoder, nexus::nn::Module, std::shared_ptr<nexus::nn::TransformerDecoder>>(nn, "TransformerDecoder")
        .def(py::init<int, int, int>(), py::arg("embed"), py::arg("heads"), py::arg("feedforward") = 0)
        .def("forward_with_memory", [](nexus::nn::TransformerDecoder& self, const nexus_model::Tensor& target, const nexus_model::Tensor& memory) {
            return self.forward(target, memory);
        })
        .def("backward", &nexus::nn::TransformerDecoder::backward);

    py::class_<nexus::nn::Residual, nexus::nn::Module, std::shared_ptr<nexus::nn::Residual>>(nn, "Residual")
        .def(py::init<std::shared_ptr<nexus::nn::Module>>())
        .def("forward", &nexus::nn::Residual::forward)
        .def("backward", &nexus::nn::Residual::backward);
    py::class_<nexus::nn::Reshape, nexus::nn::Module, std::shared_ptr<nexus::nn::Reshape>>(nn, "Reshape")
        .def(py::init<std::vector<std::size_t>>())
        .def("forward", &nexus::nn::Reshape::forward)
        .def("backward", &nexus::nn::Reshape::backward);
    nn.def("conv2d_device", [](const std::vector<float>& image, const std::vector<float>& weight, const std::vector<float>& bias, int n, int c,
                             int height, int width, int filters, int kernel_h, int kernel_w, int stride, int padding) {
        if (n < 1 || c < 1 || height < 1 || width < 1 || filters < 1 || kernel_h < 1 || kernel_w < 1 || stride < 1 || padding < 0) {
            throw std::invalid_argument("conv2d_device shape");
        }
        if (image.size() != static_cast<std::size_t>(n) * c * height * width ||
            weight.size() != static_cast<std::size_t>(filters) * c * kernel_h * kernel_w || bias.size() != static_cast<std::size_t>(filters)) {
            throw std::invalid_argument("conv2d_device length");
        }
        matrix_pro::Tensor input({static_cast<std::size_t>(n), static_cast<std::size_t>(c), static_cast<std::size_t>(height),
                                  static_cast<std::size_t>(width)},
                                 image);
        matrix_pro::Tensor weights({static_cast<std::size_t>(filters), static_cast<std::size_t>(c), static_cast<std::size_t>(kernel_h),
                                    static_cast<std::size_t>(kernel_w)},
                                   weight);
        matrix_pro::Tensor bias_tensor({static_cast<std::size_t>(filters)}, bias);
        matrix_pro::Tensor output;
        {
            py::gil_scoped_release release;
            input.upload();
            weights.upload();
            bias_tensor.upload();
            output = matrix_pro::conv2d(input, weights, bias_tensor, static_cast<std::size_t>(stride), static_cast<std::size_t>(padding));
            output.download();
        }
        py::dict out;
        out["batch"] = n;
        out["filters"] = filters;
        out["height"] = static_cast<int>(output.shape()[2]);
        out["width"] = static_cast<int>(output.shape()[3]);
        out["nchw"] = output.data();
        return out;
    }, py::arg("image"), py::arg("weight"), py::arg("bias"), py::arg("batch"), py::arg("channels"), py::arg("height"), py::arg("width"),
       py::arg("filters"), py::arg("kernel_h"), py::arg("kernel_w"), py::arg("stride") = 1, py::arg("padding") = 0);
    nn.def("swish", &nexus::nn::swish, py::arg("name") = "Swish");
    nn.def("activation", [](const std::string& name, py::function forward, py::object backward) {
        auto build = [name, forward, backward]() {
            auto fwd = [forward](float x) {
                py::gil_scoped_acquire guard;
                return forward(x).cast<float>();
            };
            if (backward.is_none()) {
                return nexus::nn::make_activation(name, fwd);
            }
            py::function bwd = py::cast<py::function>(backward);
            auto back = [bwd](float x, float grad) {
                py::gil_scoped_acquire guard;
                return bwd(x, grad).cast<float>();
            };
            return nexus::nn::make_activation(name, fwd, back);
        };
        if (name != "Swish") {
            nexus::nxs::register_named_activation(name, build);
        }
        return build();
    }, py::arg("name"), py::arg("forward"), py::arg("backward") = py::none());
    nn.def("PythonModule", [](const std::string& name, py::function forward, py::object backward,
                            const std::vector<std::vector<std::size_t>>& shapes) -> std::shared_ptr<nexus::nn::Module> {
        auto build = [name, forward, backward, shapes]() -> std::shared_ptr<nexus::nn::Module> {
            return std::make_shared<PythonLayer>(name, shapes, forward, backward);
        };
        nexus::nxs::register_named_module(name, build);
        return build();
    }, py::arg("name"), py::arg("forward"), py::arg("backward") = py::none(), py::arg("shapes") = std::vector<std::vector<std::size_t>>{});

    py::class_<nexus::nn::Sequential, nexus::nn::Module, std::shared_ptr<nexus::nn::Sequential>>(nn, "Sequential")
        .def(py::init<>())
        .def("add", [](nexus::nn::Sequential& self, std::shared_ptr<nexus::nn::Module> layer) -> nexus::nn::Sequential& {
            self.add(std::move(layer));
            return self;
        }, py::return_value_policy::reference)
        .def("forward", &nexus::nn::Sequential::forward)
        .def("backward", &nexus::nn::Sequential::backward)
        .def("train", &nexus::nn::Sequential::train)
        .def("eval", &nexus::nn::Sequential::eval)
        .def("save_nxs", [](const nexus::nn::Sequential& self, const std::string& path) {
            attach_python_payload();
            self.save_nxs(path);
        })
        .def_static("load_nxs", [](const std::string& path) {
            restore_python_payload(path);
            return nexus::nn::Sequential::load_nxs(path);
        })
        .def("size", &nexus::nn::Sequential::size);

    py::class_<nexus::nn::Graph, nexus::nn::Module, std::shared_ptr<nexus::nn::Graph>>(nn, "Graph")
        .def(py::init<>())
        .def("add", [](nexus::nn::Graph& self, const std::string& name, std::shared_ptr<nexus::nn::Module> layer) -> nexus::nn::Graph& {
            self.add(name, std::move(layer));
            return self;
        }, py::return_value_policy::reference, py::arg("name"), py::arg("layer"))
        .def("add", [](nexus::nn::Graph& self, const std::string& name, std::shared_ptr<nexus::nn::Module> layer,
                       const std::vector<std::string>& inputs, const std::string& merge) -> nexus::nn::Graph& {
            if (merge != "add" && merge != "concat") {
                throw std::invalid_argument("graph merge is add or concat");
            }
            self.add(name, std::move(layer), inputs, merge == "concat");
            return self;
        }, py::return_value_policy::reference, py::arg("name"), py::arg("layer"), py::arg("inputs"), py::arg("merge"))
        .def("concat", [](nexus::nn::Graph& self, const std::string& name, std::shared_ptr<nexus::nn::Module> layer,
                          const std::vector<std::string>& inputs) -> nexus::nn::Graph& {
            self.concat(name, std::move(layer), inputs);
            return self;
        }, py::return_value_policy::reference, py::arg("name"), py::arg("layer"), py::arg("inputs"))
        .def("forward", &nexus::nn::Graph::forward)
        .def("backward", &nexus::nn::Graph::backward)
        .def("train", &nexus::nn::Graph::train)
        .def("eval", &nexus::nn::Graph::eval)
        .def("size", &nexus::nn::Graph::size)
        .def("save_nxs", [](const nexus::nn::Graph& self, const std::string& path) { self.save_nxs(path); })
        .def_static("load_nxs", [](const std::string& path) { return nexus::nn::Graph::load_nxs(path); });

    auto loss = m.def_submodule("loss");
    py::class_<nexus::loss::CrossEntropy>(loss, "CrossEntropy")
        .def(py::init<float>(), py::arg("label_smoothing") = 0.0f)
        .def("forward_backward", [](const nexus::loss::CrossEntropy& loss_fn, const std::vector<float>& logits, const std::vector<int>& targets,
                                    int classes) {
            if (classes <= 1 || targets.empty() || logits.size() != targets.size() * static_cast<std::size_t>(classes)) {
                throw std::invalid_argument("CrossEntropy shape is invalid");
            }
            std::vector<float> grad(logits.size());
            const float value =
                loss_fn.forward_backward(logits.data(), targets.data(), static_cast<int>(targets.size()), classes, grad.data());
            py::dict out;
            out["loss"] = value;
            out["grad"] = std::move(grad);
            return out;
        });
    py::class_<nexus::loss::MSE>(loss, "MSE")
        .def(py::init<>())
        .def("forward_backward", [](const nexus::loss::MSE& mse, const std::vector<float>& pred, const std::vector<float>& target) {
            if (pred.size() != target.size() || pred.empty()) {
                throw std::invalid_argument("MSE lengths differ");
            }
            std::vector<float> grad(pred.size());
            const float value = mse.forward_backward(pred.data(), target.data(), static_cast<int>(pred.size()), grad.data());
            return py::make_tuple(value, grad);
        });

    auto optim = m.def_submodule("optim");
    py::class_<nexus::optim::AdamW, std::unique_ptr<nexus::optim::AdamW>>(optim, "AdamW")
        .def(py::init([](nexus::nn::Sequential& model, float lr, float weight_decay) {
            return std::make_unique<nexus::optim::AdamW>(model.parameters(), nexus::optim::AdamWOptions{lr, weight_decay, 0.9f, 0.999f, 1.0e-8f});
        }), py::arg("model"), py::arg("lr") = 1.0e-3f, py::arg("weight_decay") = 0.0f)
        .def("step", &nexus::optim::AdamW::step)
        .def("zero_grad", &nexus::optim::AdamW::zero_grad)
        .def("learning_rate", &nexus::optim::AdamW::learning_rate)
        .def("set_learning_rate", &nexus::optim::AdamW::set_learning_rate);
    py::class_<nexus::optim::SGD, std::unique_ptr<nexus::optim::SGD>>(optim, "SGD")
        .def(py::init([](nexus::nn::Sequential& model, float lr, float momentum, float weight_decay) {
            return std::make_unique<nexus::optim::SGD>(model.parameters(), nexus::optim::SgdOptions{lr, momentum, weight_decay});
        }), py::arg("model"), py::arg("lr") = 1.0e-2f, py::arg("momentum") = 0.0f, py::arg("weight_decay") = 0.0f)
        .def("step", &nexus::optim::SGD::step)
        .def("zero_grad", &nexus::optim::SGD::zero_grad)
        .def("learning_rate", &nexus::optim::SGD::learning_rate)
        .def("set_learning_rate", &nexus::optim::SGD::set_learning_rate);
    py::class_<nexus::optim::CosineAnnealing>(optim, "CosineAnnealing")
        .def(py::init<float, int, float>(), py::arg("base_lr"), py::arg("t_max"), py::arg("eta_min") = 0.0f)
        .def("step", &nexus::optim::CosineAnnealing::step);
    py::class_<nexus::optim::LinearWarmup>(optim, "LinearWarmup")
        .def(py::init<float, int>())
        .def("step", &nexus::optim::LinearWarmup::step);

    auto data = m.def_submodule("data");
    py::class_<nexus::data::DataLoader>(data, "DataLoader")
        .def(py::init<std::vector<float>, std::vector<int>, int, int, bool, std::uint32_t>(), py::arg("features"), py::arg("labels"),
             py::arg("features_per_sample"), py::arg("batch_size"), py::arg("shuffle") = true, py::arg("seed") = 1)
        .def("sample_count", &nexus::data::DataLoader::sample_count)
        .def("features_per_sample", &nexus::data::DataLoader::features_per_sample)
        .def("reset", &nexus::data::DataLoader::reset)
        .def("next", [](nexus::data::DataLoader& loader) {
            nexus::data::Batch batch;
            if (!loader.next(batch)) {
                return py::object(py::none());
            }
            py::dict out;
            out["features"] = std::move(batch.features);
            out["labels"] = std::move(batch.labels);
            out["batch_size"] = batch.batch_size;
            out["features_per_sample"] = batch.features_per_sample;
            return py::object(std::move(out));
        });
    data.def("read_csv", &nexus::data::read_csv);
    data.def("make_loader", &nexus::data::make_loader, py::arg("table"), py::arg("batch_size"), py::arg("shuffle") = true,
             py::arg("seed") = 1);
    py::class_<nexus::data::Table>(data, "Table")
        .def_readwrite("features", &nexus::data::Table::features)
        .def_readwrite("labels", &nexus::data::Table::labels)
        .def_readwrite("features_per_sample", &nexus::data::Table::features_per_sample);
    data.def("load_image", [](const std::string& path, int channels, int height, int width) {
        nexusdata::ImageDecodeOptions options;
        if (channels == 0) {
            options.channels = nexusdata::ImageChannels::Keep;
        } else if (channels == 1) {
            options.channels = nexusdata::ImageChannels::Gray;
        } else if (channels == 3) {
            options.channels = nexusdata::ImageChannels::RGB;
        } else if (channels == 4) {
            options.channels = nexusdata::ImageChannels::RGBA;
        } else {
            throw std::invalid_argument("load_image channels");
        }
        const nexusdata::Image image = nexusdata::load_image(path, options);
        py::dict out;
        out["channels"] = image.channels;
        if (height > 0 && width > 0) {
            std::vector<float> values(static_cast<std::size_t>(image.channels) * static_cast<std::size_t>(height) * static_cast<std::size_t>(width));
            nexusdata::simd::resize_to_chw_f32(image.data.data<std::uint8_t>(), static_cast<std::size_t>(image.height),
                                               static_cast<std::size_t>(image.width), static_cast<std::size_t>(image.channels), values.data(),
                                               static_cast<std::size_t>(height), static_cast<std::size_t>(width),
                                               nexusdata::simd::ResampleMode::Bilinear);
            out["height"] = height;
            out["width"] = width;
            out["chw"] = std::move(values);
            return out;
        }
        const nexusdata::NDArray tensor = nexusdata::image_to_tensor(image.data);
        std::vector<float> values(tensor.numel());
        if (!values.empty()) {
            std::memcpy(values.data(), tensor.data<float>(), values.size() * sizeof(float));
        }
        out["height"] = image.height;
        out["width"] = image.width;
        out["chw"] = std::move(values);
        return out;
    }, py::arg("path"), py::arg("channels") = 3, py::arg("height") = 0, py::arg("width") = 0);
    data.def("save_image", [](const std::string& path, const std::vector<float>& chw, int height, int width, int channels) {
        if (height < 1 || width < 1 || (channels != 1 && channels != 3 && channels != 4)) {
            throw std::invalid_argument("save_image shape");
        }
        const std::size_t plane = static_cast<std::size_t>(height) * static_cast<std::size_t>(width);
        const std::size_t count = plane * static_cast<std::size_t>(channels);
        if (chw.size() != count) {
            throw std::invalid_argument("save_image length");
        }
        std::vector<unsigned char> hwc(count);
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                for (int c = 0; c < channels; ++c) {
                    float value = chw[static_cast<std::size_t>(c) * plane + static_cast<std::size_t>(y) * static_cast<std::size_t>(width) +
                                      static_cast<std::size_t>(x)];
                    if (value < 0.0f) {
                        value = 0.0f;
                    }
                    if (value > 1.0f) {
                        value = 1.0f;
                    }
                    hwc[(static_cast<std::size_t>(y) * static_cast<std::size_t>(width) + static_cast<std::size_t>(x)) *
                            static_cast<std::size_t>(channels) +
                        static_cast<std::size_t>(c)] = static_cast<unsigned char>(value * 255.0f + 0.5f);
                }
            }
        }
        if (!nexus::pyimage::write_image(path, width, height, channels, hwc)) {
            throw std::runtime_error("save_image failed");
        }
    });

    auto metrics = m.def_submodule("metrics");
    py::class_<nexus::metrics::Accuracy>(metrics, "Accuracy")
        .def(py::init<>())
        .def("name", &nexus::metrics::Accuracy::name)
        .def("reset", &nexus::metrics::Accuracy::reset)
        .def("update", [](nexus::metrics::Accuracy& metric, const std::vector<float>& logits, const std::vector<int>& labels, int classes) {
            if (classes <= 1 || labels.empty() || logits.size() != labels.size() * static_cast<std::size_t>(classes)) {
                throw std::invalid_argument("Accuracy shape is invalid");
            }
            metric.update(logits.data(), labels.data(), static_cast<int>(labels.size()), classes);
        })
        .def("value", &nexus::metrics::Accuracy::value);
    py::class_<nexus::metrics::F1Score>(metrics, "F1Score")
        .def(py::init<>())
        .def("name", &nexus::metrics::F1Score::name)
        .def("reset", &nexus::metrics::F1Score::reset)
        .def("update", [](nexus::metrics::F1Score& metric, const std::vector<float>& logits, const std::vector<int>& labels, int classes) {
            if (classes <= 1 || labels.empty() || logits.size() != labels.size() * static_cast<std::size_t>(classes)) {
                throw std::invalid_argument("F1Score shape is invalid");
            }
            metric.update(logits.data(), labels.data(), static_cast<int>(labels.size()), classes);
        })
        .def("value", &nexus::metrics::F1Score::value);

    auto train = m.def_submodule("train");
    py::class_<nexus::EpochStats>(train, "EpochStats")
        .def_readwrite("epoch", &nexus::EpochStats::epoch)
        .def_readwrite("train_loss", &nexus::EpochStats::train_loss)
        .def_readwrite("val_loss", &nexus::EpochStats::val_loss)
        .def_readwrite("accuracy", &nexus::EpochStats::accuracy)
        .def_readwrite("f1", &nexus::EpochStats::f1);
    py::class_<nexus::callbacks::Callback, PyCallback, std::shared_ptr<nexus::callbacks::Callback>>(train, "Callback").def(py::init<>());
    py::class_<nexus::callbacks::EarlyStopping, nexus::callbacks::Callback, std::shared_ptr<nexus::callbacks::EarlyStopping>>(train, "EarlyStopping")
        .def(py::init<int, float>(), py::arg("patience"), py::arg("min_delta") = 0.0f);
    py::class_<nexus::callbacks::ProgressLogger, nexus::callbacks::Callback, std::shared_ptr<nexus::callbacks::ProgressLogger>>(train, "ProgressLogger")
        .def(py::init<>())
        .def("last", [](const nexus::callbacks::ProgressLogger& logger) {
            const nexus::callbacks::EpochState& state = logger.last();
            py::dict out;
            out["epoch"] = state.epoch;
            out["train_loss"] = state.train_loss;
            out["val_loss"] = state.val_loss;
            out["accuracy"] = state.accuracy;
            out["f1"] = state.f1;
            return out;
        });
    py::class_<nexus::LanguageTrainer>(train, "LanguageTrainer")
        .def(py::init([](nexus::nn::Module& model, float learning_rate) {
            return std::make_unique<nexus::LanguageTrainer>(model, learning_rate);
        }), py::keep_alive<1, 2>(), py::arg("model"), py::arg("learning_rate"))
        .def("fit", [](nexus::LanguageTrainer& self, const std::vector<std::vector<int>>& tokens, int epochs) {
            if (tokens.empty() || tokens.front().size() < 2) {
                throw std::invalid_argument("language tokens must be [batch, time]");
            }
            nexus::TokenRows rows;
            rows.batch = static_cast<int>(tokens.size());
            rows.time = static_cast<int>(tokens.front().size());
            rows.values.reserve(static_cast<std::size_t>(rows.batch * rows.time));
            for (const std::vector<int>& row : tokens) {
                if (static_cast<int>(row.size()) != rows.time) {
                    throw std::invalid_argument("language token rows must share one length");
                }
                rows.values.insert(rows.values.end(), row.begin(), row.end());
            }
            py::gil_scoped_release release;
            return self.fit(rows, epochs);
        }, py::arg("tokens"), py::arg("epochs"));

    py::class_<nexus::TrainerBuilder>(train, "TrainerBuilder")
        .def(py::init<nexus::nn::Sequential>())
        .def("adamw", [](nexus::TrainerBuilder& self, nexus::nn::Sequential& model, float lr, float weight_decay) -> nexus::TrainerBuilder& {
            self.optimizer(nexus::optim::AdamW(model.parameters(), nexus::optim::AdamWOptions{lr, weight_decay, 0.9f, 0.999f, 1.0e-8f}));
            return self;
        }, py::return_value_policy::reference, py::arg("model"), py::arg("lr") = 1.0e-3f, py::arg("weight_decay") = 0.0f)
        .def("sgd", [](nexus::TrainerBuilder& self, nexus::nn::Sequential& model, float lr, float momentum, float weight_decay) -> nexus::TrainerBuilder& {
            self.optimizer(nexus::optim::SGD(model.parameters(), nexus::optim::SgdOptions{lr, momentum, weight_decay}));
            return self;
        }, py::return_value_policy::reference, py::arg("model"), py::arg("lr") = 1.0e-2f, py::arg("momentum") = 0.0f, py::arg("weight_decay") = 0.0f)
        .def("loss", [](nexus::TrainerBuilder& self, float smoothing) -> nexus::TrainerBuilder& {
            self.loss(nexus::loss::CrossEntropy(smoothing));
            return self;
        }, py::return_value_policy::reference, py::arg("label_smoothing") = 0.0f)
        .def("execution_policy_hybrid", [](nexus::TrainerBuilder& self) -> nexus::TrainerBuilder& {
            self.execution_policy(nexus::ExecutionPolicy::hybrid_cpu_multi_gpu());
            return self;
        }, py::return_value_policy::reference)
        .def("execution_policy_cpu", [](nexus::TrainerBuilder& self) -> nexus::TrainerBuilder& {
            self.execution_policy(nexus::ExecutionPolicy::cpu());
            return self;
        }, py::return_value_policy::reference)
        .def("gradient_clip_norm", [](nexus::TrainerBuilder& self, float norm) -> nexus::TrainerBuilder& {
            self.gradient_clip_norm(norm);
            return self;
        }, py::return_value_policy::reference)
        .def("accuracy", [](nexus::TrainerBuilder& self) -> nexus::TrainerBuilder& {
            self.metric<nexus::metrics::Accuracy>();
            return self;
        }, py::return_value_policy::reference)
        .def("f1", [](nexus::TrainerBuilder& self) -> nexus::TrainerBuilder& {
            self.metric<nexus::metrics::F1Score>();
            return self;
        }, py::return_value_policy::reference)
        .def("mixed_precision", [](nexus::TrainerBuilder& self, bool enabled) -> nexus::TrainerBuilder& {
            self.mixed_precision(enabled);
            return self;
        }, py::return_value_policy::reference, py::arg("enabled") = true)
        .def("callback", [](nexus::TrainerBuilder& self, std::shared_ptr<nexus::callbacks::Callback> callback) -> nexus::TrainerBuilder& {
            self.callback(std::move(callback));
            return self;
        }, py::return_value_policy::reference)
        .def("cosine", [](nexus::TrainerBuilder& self, nexus::optim::CosineAnnealing scheduler) -> nexus::TrainerBuilder& {
            self.learning_rate_schedule(std::move(scheduler));
            return self;
        }, py::return_value_policy::reference)
        .def("warmup", [](nexus::TrainerBuilder& self, nexus::optim::LinearWarmup scheduler) -> nexus::TrainerBuilder& {
            self.learning_rate_schedule(std::move(scheduler));
            return self;
        }, py::return_value_policy::reference)
        .def("early_stopping", [](nexus::TrainerBuilder& self, int patience) -> nexus::TrainerBuilder& {
            self.callback<nexus::callbacks::EarlyStopping>(patience);
            return self;
        }, py::return_value_policy::reference)
        .def("build", [](nexus::TrainerBuilder& self) { return std::make_unique<nexus::Trainer>(self.build()); });
    py::class_<nexus::Trainer, std::unique_ptr<nexus::Trainer>>(train, "Trainer")
        .def("fit", [](nexus::Trainer& self, nexus::data::DataLoader& train_loader, nexus::data::DataLoader& validation, int epochs) {
            py::gil_scoped_release release;
            self.fit(train_loader, validation, epochs);
        })
        .def("history", &nexus::Trainer::history)
        .def("epochs_ran", &nexus::Trainer::epochs_ran)
        .def("save_nxs", [](const nexus::Trainer& self, const std::string& path) {
            attach_python_payload();
            self.save_nxs(path);
        })
        .def("load_nxs", &nexus::Trainer::load_nxs)
        .def("model", &nexus::Trainer::model, py::return_value_policy::reference);
    train.def("all_reduce_mean", [](std::vector<std::vector<float>> ranks) {
        nexus::train::all_reduce_mean(ranks);
        return ranks;
    });

    auto rl = m.def_submodule("rl");
    py::class_<nexus::rl::Environment, PyEnvironment, std::shared_ptr<nexus::rl::Environment>>(rl, "Environment")
        .def(py::init<int, int, bool>(), py::arg("observation_dim"), py::arg("actions"), py::arg("discrete") = true)
        .def("observation_dim", &nexus::rl::Environment::observation_dim)
        .def("action_count", &nexus::rl::Environment::action_count)
        .def("reset", [](nexus::rl::Environment& env, std::uint32_t seed) { return host_floats(env.reset(seed)); })
        .def("step", [](nexus::rl::Environment& env, py::object action) {
            if (py::isinstance<py::list>(action) || py::isinstance<py::tuple>(action)) {
                const auto values = action.cast<std::vector<float>>();
                return step_dict(env.step(nexus::Tensor(nexus::Shape{values.size()}, std::span<const float>(values.data(), values.size()))));
            }
            return step_dict(env.step(nexus::rl::discrete_action_tensor(action.cast<int>())));
        });
    bind_env<nexus::rl::LineWorld>(rl, "LineWorld");
    bind_env<nexus::rl::GridWorld>(rl, "GridWorld");
    bind_env<nexus::rl::CartPole>(rl, "CartPole");
    bind_env<nexus::rl::PointMass>(rl, "PointMass");

    py::class_<nexus::rl::PPOAgent::Config>(rl, "PPOConfig")
        .def(py::init<>())
        .def_readwrite("hidden", &nexus::rl::PPOAgent::Config::hidden)
        .def_readwrite("gamma", &nexus::rl::PPOAgent::Config::gamma)
        .def_readwrite("lambda_", &nexus::rl::PPOAgent::Config::lambda)
        .def_readwrite("clip_epsilon", &nexus::rl::PPOAgent::Config::clip_epsilon)
        .def_readwrite("learning_rate", &nexus::rl::PPOAgent::Config::learning_rate)
        .def_readwrite("entropy_coef", &nexus::rl::PPOAgent::Config::entropy_coef)
        .def_readwrite("value_coef", &nexus::rl::PPOAgent::Config::value_coef)
        .def_readwrite("epochs", &nexus::rl::PPOAgent::Config::epochs)
        .def_readwrite("seed", &nexus::rl::PPOAgent::Config::seed);
    py::class_<nexus::rl::PPOAgent>(rl, "PPOAgent")
        .def(py::init<int, int, nexus::rl::PPOAgent::Config>(), py::arg("observation_dim"), py::arg("actions"),
             py::arg("config") = nexus::rl::PPOAgent::Config{})
        .def("learn", [](nexus::rl::PPOAgent& agent, std::shared_ptr<nexus::rl::Environment> env, int iterations, int horizon) {
            py::gil_scoped_release release;
            return agent.learn(*env, iterations, horizon);
        })
        .def("act", [](nexus::rl::PPOAgent& agent, const std::vector<float>& observation, bool greedy) {
            float log_prob = 0.0f;
            float value = 0.0f;
            const int action = agent.act(observation.data(), log_prob, value, greedy);
            return py::make_tuple(action, log_prob, value);
        })
        .def("save_nxs", &nexus::rl::PPOAgent::save_nxs)
        .def_static("load_nxs", &nexus::rl::PPOAgent::load_nxs);
    py::class_<nexus::rl::DQNAgent::Config>(rl, "DQNConfig").def(py::init<>()).def_readwrite("hidden", &nexus::rl::DQNAgent::Config::hidden)
        .def_readwrite("seed", &nexus::rl::DQNAgent::Config::seed)
        .def_readwrite("dueling", &nexus::rl::DQNAgent::Config::dueling);
    py::class_<nexus::rl::DQNAgent>(rl, "DQNAgent")
        .def(py::init<int, int, nexus::rl::DQNAgent::Config>(), py::arg("observation_dim"), py::arg("actions"),
             py::arg("config") = nexus::rl::DQNAgent::Config{})
        .def("learn", [](nexus::rl::DQNAgent& agent, std::shared_ptr<nexus::rl::Environment> env, int steps) {
            py::gil_scoped_release release;
            return agent.learn(*env, steps);
        })
        .def("act", [](nexus::rl::DQNAgent& agent, const std::vector<float>& observation, bool greedy) {
            return agent.act(observation.data(), greedy);
        })
        .def("save_nxs", &nexus::rl::DQNAgent::save_nxs)
        .def_static("load_nxs", &nexus::rl::DQNAgent::load_nxs);
    py::class_<nexus::rl::SACAgent::Config>(rl, "SACConfig").def(py::init<>()).def_readwrite("hidden", &nexus::rl::SACAgent::Config::hidden);
    py::class_<nexus::rl::SACAgent>(rl, "SACAgent")
        .def(py::init<int, nexus::rl::SACAgent::Config>(), py::arg("observation_dim"), py::arg("config") = nexus::rl::SACAgent::Config{})
        .def("learn", [](nexus::rl::SACAgent& agent, std::shared_ptr<nexus::rl::Environment> env, int steps) {
            py::gil_scoped_release release;
            return agent.learn(*env, steps);
        })
        .def("act", [](nexus::rl::SACAgent& agent, const std::vector<float>& observation, bool greedy) {
            return agent.act(observation.data(), greedy);
        })
        .def("save_nxs", &nexus::rl::SACAgent::save_nxs)
        .def_static("load_nxs", &nexus::rl::SACAgent::load_nxs);
    py::class_<nexus::rl::ActorCriticAgent>(rl, "ActorCriticAgent")
        .def_static("discrete", &nexus::rl::ActorCriticAgent::discrete)
        .def("learn", [](nexus::rl::ActorCriticAgent& agent, std::shared_ptr<nexus::rl::Environment> env, int iterations, int horizon) {
            py::gil_scoped_release release;
            return agent.learn(*env, iterations, horizon);
        })
        .def("act", [](nexus::rl::ActorCriticAgent& agent, const std::vector<float>& observation, bool greedy) {
            float log_prob = 0.0f;
            float value = 0.0f;
            const int action = agent.act(observation.data(), log_prob, value, greedy);
            return py::make_tuple(action, log_prob, value);
        })
        .def("save_nxs", &nexus::rl::ActorCriticAgent::save_nxs)
        .def_static("load_nxs", &nexus::rl::ActorCriticAgent::load_nxs);
    py::class_<nexus::rl::ContinuousActorCritic>(rl, "ContinuousActorCritic")
        .def(py::init<int, std::uint32_t>())
        .def("act", [](nexus::rl::ContinuousActorCritic& agent, const std::vector<float>& observation, bool greedy) {
            return agent.act(observation.data(), greedy);
        })
        .def("learn", [](nexus::rl::ContinuousActorCritic& agent, std::shared_ptr<nexus::rl::Environment> env, int iterations, int horizon) {
            py::gil_scoped_release release;
            return agent.learn(*env, iterations, horizon);
        })
        .def("save_nxs", &nexus::rl::ContinuousActorCritic::save_nxs)
        .def_static("load_nxs", &nexus::rl::ContinuousActorCritic::load_nxs);
    py::class_<nexus::rl::League>(rl, "League")
        .def(py::init<>())
        .def("add_player", &nexus::rl::League::add_player, py::arg("elo") = 1000.0f)
        .def("save", &nexus::rl::League::save)
        .def("weights", &nexus::rl::League::weights)
        .def("elo", &nexus::rl::League::elo)
        .def("result", &nexus::rl::League::result)
        .def("size", &nexus::rl::League::size)
        .def("save_nxs", &nexus::rl::League::save_nxs)
        .def_static("load_nxs", &nexus::rl::League::load_nxs);
    py::class_<nexus::rl::ReplayBuffer>(rl, "ReplayBuffer")
        .def(py::init<int, int>())
        .def("size", &nexus::rl::ReplayBuffer::size)
        .def("add", [](nexus::rl::ReplayBuffer& buffer, const std::vector<float>& observation, int action, float reward,
                       const std::vector<float>& next_observation, bool terminated) {
            buffer.add(observation.data(), action, reward, next_observation.data(), terminated);
        })
        .def("sample", [](const nexus::rl::ReplayBuffer& buffer, const std::vector<int>& indices) {
            const int dim = buffer.observation_dim();
            std::vector<float> observations(indices.size() * static_cast<std::size_t>(dim));
            std::vector<float> next_observations(indices.size() * static_cast<std::size_t>(dim));
            std::vector<int> actions(indices.size());
            std::vector<float> rewards(indices.size());
            std::vector<std::uint8_t> terminated(indices.size());
            buffer.sample(indices, observations.data(), next_observations.data(), actions.data(), rewards.data(), terminated.data());
            py::dict out;
            out["observations"] = std::move(observations);
            out["next_observations"] = std::move(next_observations);
            out["actions"] = std::move(actions);
            out["rewards"] = std::move(rewards);
            out["terminated"] = std::move(terminated);
            return out;
        });
    py::class_<nexus::rl::PrioritizedReplayBuffer>(rl, "PrioritizedReplayBuffer")
        .def(py::init<int, int, float>())
        .def("size", &nexus::rl::PrioritizedReplayBuffer::size)
        .def("add", [](nexus::rl::PrioritizedReplayBuffer& buffer, const std::vector<float>& observation, int action, float reward,
                       const std::vector<float>& next_observation, bool terminated, float priority) {
            buffer.add(observation.data(), action, reward, next_observation.data(), terminated, priority);
        })
        .def("sample", [](const nexus::rl::PrioritizedReplayBuffer& buffer, int batch, float beta, std::uint32_t seed) {
            nexus::rl::Rng rng(seed);
            const auto sample = buffer.sample(batch, beta, rng);
            py::dict out;
            out["indices"] = sample.indices;
            out["weights"] = sample.weights;
            out["observations"] = sample.observations;
            out["next_observations"] = sample.next_observations;
            out["actions"] = sample.actions;
            out["rewards"] = sample.rewards;
            out["terminated"] = sample.terminated;
            return out;
        });
    py::class_<nexus::rl::RolloutBuffer>(rl, "RolloutBuffer")
        .def(py::init<>())
        .def("reset", &nexus::rl::RolloutBuffer::reset)
        .def("add", [](nexus::rl::RolloutBuffer& buffer, const std::vector<float>& observation, int action, float log_prob, float value,
                       float reward, bool terminated) { buffer.add(observation.data(), action, log_prob, value, reward, terminated); })
        .def("finish", &nexus::rl::RolloutBuffer::finish)
        .def("size", &nexus::rl::RolloutBuffer::size)
        .def("actions", [](const nexus::rl::RolloutBuffer& buffer) { return buffer.actions(); })
        .def("log_probs", [](const nexus::rl::RolloutBuffer& buffer) { return buffer.log_probs(); })
        .def("values", [](const nexus::rl::RolloutBuffer& buffer) { return buffer.values(); })
        .def("advantages", [](const nexus::rl::RolloutBuffer& buffer) { return buffer.advantages(); })
        .def("returns", [](const nexus::rl::RolloutBuffer& buffer) { return buffer.returns(); });
    py::class_<nexus::rl::VectorEnv, std::unique_ptr<nexus::rl::VectorEnv>>(rl, "VectorEnv")
        .def_static("line_worlds", [](int count) {
            std::vector<std::unique_ptr<nexus::rl::Environment>> envs;
            for (int i = 0; i < count; ++i) {
                envs.push_back(std::make_unique<nexus::rl::LineWorld>());
            }
            return std::make_unique<nexus::rl::VectorEnv>(std::move(envs));
        })
        .def("reset", [](nexus::rl::VectorEnv& env, std::uint32_t seed) { return host_floats(env.reset(seed)); })
        .def("step", [](nexus::rl::VectorEnv& env, const std::vector<int>& actions) {
            const nexus::rl::VectorStep step = env.step(actions);
            py::dict out;
            out["observations"] = host_floats(step.observations);
            out["rewards"] = step.rewards;
            out["terminated"] = step.terminated;
            return out;
        });
    py::class_<nexus::rl::MarlReport>(rl, "MarlReport")
        .def_readwrite("first_return", &nexus::rl::MarlReport::first_return)
        .def_readwrite("last_return", &nexus::rl::MarlReport::last_return);
    py::class_<nexus::rl::MultiAgentTrainer>(rl, "MultiAgentTrainer")
        .def(py::init<>())
        .def("train_independent", [](nexus::rl::MultiAgentTrainer& trainer, nexus::rl::ButtonWorld& env, int iterations, int horizon,
                                     std::uint32_t seed) {
            py::gil_scoped_release release;
            return trainer.train_independent(env, iterations, horizon, seed);
        });
    py::class_<nexus::rl::ButtonWorld, std::shared_ptr<nexus::rl::ButtonWorld>>(rl, "ButtonWorld").def(py::init<>());
    py::class_<nexus::rl::TeamButton, std::shared_ptr<nexus::rl::TeamButton>>(rl, "TeamButton")
        .def(py::init([](int agents, int horizon) { return nexus::rl::TeamButton(agents, horizon); }), py::arg("agents") = 2, py::arg("horizon") = 6);
    py::class_<nexus::rl::Duel, std::shared_ptr<nexus::rl::Duel>>(rl, "Duel").def(py::init<>());
    py::class_<nexus::rl::TurnLine, std::shared_ptr<nexus::rl::TurnLine>>(rl, "TurnLine").def(py::init<>());

    auto custom = m.def_submodule("custom");
    py::class_<nexus::MicroTape>(custom, "MicroTape")
        .def(py::init<>())
        .def("constant", &nexus::MicroTape::constant)
        .def("input", &nexus::MicroTape::input)
        .def("add", &nexus::MicroTape::add)
        .def("sub", &nexus::MicroTape::sub)
        .def("mul", &nexus::MicroTape::mul)
        .def("div", &nexus::MicroTape::div)
        .def("neg", &nexus::MicroTape::neg)
        .def("square", &nexus::MicroTape::square)
        .def("exp", &nexus::MicroTape::exp)
        .def("log", &nexus::MicroTape::log)
        .def("abs", &nexus::MicroTape::abs)
        .def("max", &nexus::MicroTape::max)
        .def("backward", &nexus::MicroTape::backward)
        .def("value", &nexus::MicroTape::value)
        .def("grad", &nexus::MicroTape::grad)
        .def("reset", &nexus::MicroTape::reset)
        .def("size", &nexus::MicroTape::size);
    py::class_<nexus::Tape>(custom, "Tape")
        .def(py::init<>())
        .def("input", [](nexus::Tape& tape, const nexus_model::Tensor& value) { return tape.input(value); })
        .def("add", &nexus::Tape::add)
        .def("mul", &nexus::Tape::mul)
        .def("matmul", &nexus::Tape::matmul)
        .def("relu", &nexus::Tape::relu)
        .def("apply", [](nexus::Tape& tape, std::shared_ptr<nexus::nn::Module> module, int source) { return tape.apply(std::move(module), source); })
        .def("backward", &nexus::Tape::backward)
        .def("value", &nexus::Tape::value)
        .def("grad", &nexus::Tape::grad);
    py::class_<nexus::MicroTape::Var>(custom, "Var");
    py::class_<nexus::CustomMetric>(custom, "Metric")
        .def(py::init([](const std::string& name, py::function fn) {
            return nexus::CustomMetric(name, [fn](std::span<const float> pred, std::span<const float> target) {
                py::gil_scoped_acquire guard;
                return fn(std::vector<float>(pred.begin(), pred.end()), std::vector<float>(target.begin(), target.end())).cast<double>();
            });
        }))
        .def("name", &nexus::CustomMetric::name)
        .def("update", [](nexus::CustomMetric& metric, const std::vector<float>& pred, const std::vector<float>& target) {
            metric.update(pred, target);
        })
        .def("value", &nexus::CustomMetric::value)
        .def("reset", &nexus::CustomMetric::reset);
    py::class_<nexus::CustomLoss>(custom, "Loss")
        .def_static("numeric", [](const std::string& name, py::function fn) {
            return nexus::CustomLoss::from_numeric(name, [fn](std::span<const float> pred, std::span<const float> target) {
                py::gil_scoped_acquire guard;
                return fn(std::vector<float>(pred.begin(), pred.end()), std::vector<float>(target.begin(), target.end())).cast<double>();
            });
        })
        .def("name", &nexus::CustomLoss::name);
    py::class_<nexus::CustomOptimizer>(custom, "Optimizer")
        .def(py::init([](nexus::nn::Sequential& model, float learning_rate, py::function rule) {
            return nexus::CustomOptimizer(model.parameters(), learning_rate,
                                         [rule](float* param, const float* grad, std::size_t n, float lr) {
                                             std::vector<float> values(param, param + n);
                                             std::vector<float> grads(grad, grad + n);
                                             py::gil_scoped_acquire guard;
                                             py::list updated = rule(values, grads, lr);
                                             if (static_cast<std::size_t>(updated.size()) != n) {
                                                 throw std::invalid_argument("custom optimizer rule length mismatch");
                                             }
                                             for (std::size_t i = 0; i < n; ++i) {
                                                 param[i] = updated[i].cast<float>();
                                             }
                                         });
        }))
        .def("step", &nexus::CustomOptimizer::step)
        .def("zero_grad", &nexus::CustomOptimizer::zero_grad)
        .def("learning_rate", &nexus::CustomOptimizer::learning_rate)
        .def("set_learning_rate", &nexus::CustomOptimizer::set_learning_rate);

    auto linalg = m.def_submodule("linalg");
    linalg.def("kernel_name", [] { return std::string(matrix_pro::gemm_kernel_name()); });
    linalg.def("gemm", [](int m, int n, int k, const std::vector<float>& a, const std::vector<float>& b, float alpha, float beta) {
        if (a.size() != static_cast<std::size_t>(m) * static_cast<std::size_t>(k) ||
            b.size() != static_cast<std::size_t>(k) * static_cast<std::size_t>(n)) {
            throw std::invalid_argument("gemm shapes do not match");
        }
        std::vector<float> c(static_cast<std::size_t>(m) * static_cast<std::size_t>(n), 0.0f);
        {
            py::gil_scoped_release release;
            matrix_pro::gemm_f32(m, n, k, a.data(), k, b.data(), n, c.data(), n, alpha, beta);
        }
        return c;
    }, py::arg("m"), py::arg("n"), py::arg("k"), py::arg("a"), py::arg("b"), py::arg("alpha") = 1.0f, py::arg("beta") = 0.0f);
}
