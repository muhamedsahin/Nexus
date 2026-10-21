#pragma once

/**
 * @file modules.hpp
 * @brief nexus_model katmanlarinin akici kabugu.
 * `Sequential` kopyasi katmanlari paylasir. `model->parameters()` bu yuzden calisir.
 */

#include "nexus/custom/activation.hpp"
#include "nexus_model/activations/activations.hpp"
#include "nexus_model/layers/attention.hpp"
#include "nexus_model/layers/conv.hpp"
#include "nexus_model/layers/dropout.hpp"
#include "nexus_model/layers/embedding.hpp"
#include "nexus_model/layers/flatten_reshape.hpp"
#include "nexus_model/layers/linear.hpp"
#include "nexus_model/layers/normalization.hpp"
#include "nexus_model/layers/pooling.hpp"
#include "nexus_model/layers/recurrent.hpp"
#include "nexus_model/core/kernels.hpp"
#include "nexus/model/nxs_format.hpp"

#include <algorithm>
#include <functional>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace nexus {
namespace nn {

class Module {
 public:
    virtual ~Module() = default;
    virtual nexus_model::Tensor forward(const nexus_model::Tensor& input) = 0;
    virtual nexus_model::Tensor backward(const nexus_model::Tensor& grad_output) = 0;
    [[nodiscard]] virtual std::vector<nexus_model::Parameter*> parameters() const = 0;
    virtual void zero_grad() {
        for (nexus_model::Parameter* parameter : parameters()) {
            parameter->zero_grad();
        }
    }
    virtual void train() {}
    virtual void eval() {}
    [[nodiscard]] virtual nxs::LayerSpec nxs_spec() const {
        throw std::logic_error("bu katmanin govdesi dosyaya yazilmaz");
    }
};

class Linear : public Module {
 public:
    Linear(std::size_t in_features, std::size_t out_features, bool use_bias = true)
        : in_(in_features), out_(out_features), bias_(use_bias), impl_(in_features, out_features, use_bias) {}
    void train() override { impl_.train(); }
    void eval() override { impl_.eval(); }
    nexus_model::Tensor forward(const nexus_model::Tensor& input) override { return impl_.forward(input); }
    nexus_model::Tensor backward(const nexus_model::Tensor& grad_output) override { return impl_.backward(grad_output); }
    [[nodiscard]] std::vector<nexus_model::Parameter*> parameters() const override { return impl_.parameters(); }
    void zero_grad() override { impl_.zero_grad(); }
    [[nodiscard]] nxs::LayerSpec nxs_spec() const override {
        return nxs::spec_of(nxs::LayerKind::Linear, "Linear",
                            {nxs::i32("in", static_cast<int>(in_)), nxs::i32("out", static_cast<int>(out_)), nxs::boolean("bias", bias_)});
    }

 private:
    std::size_t in_ = 0;
    std::size_t out_ = 0;
    bool bias_ = true;
    mutable nexus_model::Linear impl_;
};

class Conv2d : public Module {
 public:
    Conv2d(int in_channels, int out_channels, int kernel, int stride = 1, int padding = 0, bool bias = true)
        : in_(in_channels), out_(out_channels), kernel_(kernel), stride_(stride), padding_(padding), bias_(bias),
          impl_(in_channels, out_channels, kernel, stride, padding, 1, 1, bias) {}
    void train() override { impl_.train(); }
    void eval() override { impl_.eval(); }
    nexus_model::Tensor forward(const nexus_model::Tensor& input) override { return impl_.forward(input); }
    nexus_model::Tensor backward(const nexus_model::Tensor& grad_output) override { return impl_.backward(grad_output); }
    [[nodiscard]] std::vector<nexus_model::Parameter*> parameters() const override { return impl_.parameters(); }
    void zero_grad() override { impl_.zero_grad(); }
    [[nodiscard]] nxs::LayerSpec nxs_spec() const override {
        return nxs::spec_of(nxs::LayerKind::Conv2d, "Conv2d",
                            {nxs::i32("in", in_), nxs::i32("out", out_), nxs::i32("kernel", kernel_), nxs::i32("stride", stride_),
                             nxs::i32("padding", padding_), nxs::boolean("bias", bias_)});
    }

 private:
    int in_ = 0;
    int out_ = 0;
    int kernel_ = 1;
    int stride_ = 1;
    int padding_ = 0;
    bool bias_ = true;
    mutable nexus_model::Conv2D impl_;
};

class MultiHeadAttention : public Module {
 public:
    MultiHeadAttention(int embed, int heads) : embed_(embed), heads_(heads), impl_(embed, heads, 0.0f, false) {}
    void train() override { impl_.train(); }
    void eval() override { impl_.eval(); }
    nexus_model::Tensor forward(const nexus_model::Tensor& input) override { return impl_.forward(input); }
    nexus_model::Tensor backward(const nexus_model::Tensor& grad_output) override { return impl_.backward(grad_output); }
    [[nodiscard]] std::vector<nexus_model::Parameter*> parameters() const override { return impl_.parameters(); }
    void zero_grad() override { impl_.zero_grad(); }
    [[nodiscard]] nxs::LayerSpec nxs_spec() const override {
        return nxs::spec_of(nxs::LayerKind::MultiHeadAttention, "MultiHeadAttention",
                            {nxs::i32("embed", embed_), nxs::i32("heads", heads_)});
    }

 private:
    int embed_ = 0;
    int heads_ = 1;
    mutable nexus_model::MultiHeadAttention impl_;
};

class LSTM : public Module {
 public:
    LSTM(int input_size, int hidden_size) : input_(input_size), hidden_(hidden_size), impl_(input_size, hidden_size) {}
    void train() override { impl_.train(); }
    void eval() override { impl_.eval(); }
    nexus_model::Tensor forward(const nexus_model::Tensor& input) override { return impl_.forward(input); }
    nexus_model::Tensor backward(const nexus_model::Tensor& grad_output) override { return impl_.backward(grad_output); }
    [[nodiscard]] std::vector<nexus_model::Parameter*> parameters() const override { return impl_.parameters(); }
    void zero_grad() override { impl_.zero_grad(); }
    [[nodiscard]] nxs::LayerSpec nxs_spec() const override {
        return nxs::spec_of(nxs::LayerKind::Lstm, "LSTM", {nxs::i32("input", input_), nxs::i32("hidden", hidden_)});
    }

 private:
    int input_ = 0;
    int hidden_ = 0;
    mutable nexus_model::LSTM impl_;
};

class GRU : public Module {
 public:
    GRU(int input_size, int hidden_size) : input_(input_size), hidden_(hidden_size), impl_(input_size, hidden_size) {}
    void train() override { impl_.train(); }
    void eval() override { impl_.eval(); }
    nexus_model::Tensor forward(const nexus_model::Tensor& input) override { return impl_.forward(input); }
    nexus_model::Tensor backward(const nexus_model::Tensor& grad_output) override { return impl_.backward(grad_output); }
    [[nodiscard]] std::vector<nexus_model::Parameter*> parameters() const override { return impl_.parameters(); }
    void zero_grad() override { impl_.zero_grad(); }
    [[nodiscard]] nxs::LayerSpec nxs_spec() const override {
        return nxs::spec_of(nxs::LayerKind::Gru, "GRU", {nxs::i32("input", input_), nxs::i32("hidden", hidden_)});
    }

 private:
    int input_ = 0;
    int hidden_ = 0;
    mutable nexus_model::GRU impl_;
};

class TransformerEncoder : public Module {
 public:
    TransformerEncoder(int embed, int heads, int feedforward = 0)
        : embed_(embed), heads_(heads), feedforward_(feedforward), impl_(embed, heads, feedforward, 0.0f) {}
    void train() override { impl_.train(); }
    void eval() override { impl_.eval(); }
    nexus_model::Tensor forward(const nexus_model::Tensor& input) override { return impl_.forward(input); }
    nexus_model::Tensor backward(const nexus_model::Tensor& grad_output) override { return impl_.backward(grad_output); }
    [[nodiscard]] std::vector<nexus_model::Parameter*> parameters() const override { return impl_.parameters(); }
    void zero_grad() override { impl_.zero_grad(); }
    [[nodiscard]] nxs::LayerSpec nxs_spec() const override {
        return nxs::spec_of(nxs::LayerKind::TransformerEncoder, "TransformerEncoder",
                            {nxs::i32("embed", embed_), nxs::i32("heads", heads_), nxs::i32("feedforward", feedforward_)});
    }

 private:
    int embed_ = 0;
    int heads_ = 1;
    int feedforward_ = 0;
    mutable nexus_model::TransformerEncoderLayer impl_;
};

class Embedding : public Module {
 public:
    Embedding(int num_embeddings, int embedding_dim, int padding_idx = -1)
        : num_(num_embeddings), dim_(embedding_dim), padding_(padding_idx), impl_(num_embeddings, embedding_dim, padding_idx) {}
    void train() override { impl_.train(); }
    void eval() override { impl_.eval(); }
    nexus_model::Tensor forward(const nexus_model::Tensor& input) override { return impl_.forward(input); }
    nexus_model::Tensor backward(const nexus_model::Tensor& grad_output) override { return impl_.backward(grad_output); }
    [[nodiscard]] std::vector<nexus_model::Parameter*> parameters() const override { return impl_.parameters(); }
    void zero_grad() override { impl_.zero_grad(); }
    [[nodiscard]] nxs::LayerSpec nxs_spec() const override {
        return nxs::spec_of(nxs::LayerKind::Embedding, "Embedding",
                            {nxs::i32("num", num_), nxs::i32("dim", dim_), nxs::i32("padding", padding_)});
    }

 private:
    int num_ = 0;
    int dim_ = 0;
    int padding_ = -1;
    mutable nexus_model::Embedding impl_;
};

class BatchNorm1d : public Module {
 public:
    explicit BatchNorm1d(int num_features) : features_(num_features), impl_(num_features) {}
    void train() override { impl_.train(); }
    void eval() override { impl_.eval(); }
    nexus_model::Tensor forward(const nexus_model::Tensor& input) override { return impl_.forward(input); }
    nexus_model::Tensor backward(const nexus_model::Tensor& grad_output) override { return impl_.backward(grad_output); }
    [[nodiscard]] std::vector<nexus_model::Parameter*> parameters() const override { return impl_.parameters(); }
    void zero_grad() override { impl_.zero_grad(); }
    [[nodiscard]] nxs::LayerSpec nxs_spec() const override {
        return nxs::spec_of(nxs::LayerKind::BatchNorm1d, "BatchNorm1d", {nxs::i32("features", features_)});
    }

 private:
    int features_ = 0;
    mutable nexus_model::BatchNorm1D impl_;
};

class LayerNorm : public Module {
 public:
    explicit LayerNorm(std::size_t normalized_size, float eps = 1.0e-5f)
        : size_(normalized_size), eps_(eps), impl_(normalized_size, eps) {}
    void train() override { impl_.train(); }
    void eval() override { impl_.eval(); }
    nexus_model::Tensor forward(const nexus_model::Tensor& input) override { return impl_.forward(input); }
    nexus_model::Tensor backward(const nexus_model::Tensor& grad_output) override { return impl_.backward(grad_output); }
    [[nodiscard]] std::vector<nexus_model::Parameter*> parameters() const override { return impl_.parameters(); }
    void zero_grad() override { impl_.zero_grad(); }
    [[nodiscard]] nxs::LayerSpec nxs_spec() const override {
        return nxs::spec_of(nxs::LayerKind::LayerNorm, "LayerNorm",
                            {nxs::i32("size", static_cast<int>(size_)), nxs::f32("eps", eps_)});
    }

 private:
    std::size_t size_ = 0;
    float eps_ = 1.0e-5f;
    mutable nexus_model::LayerNorm impl_;
};

class ReLU : public Module {
 public:
    void train() override { impl_.train(); }
    void eval() override { impl_.eval(); }
    nexus_model::Tensor forward(const nexus_model::Tensor& input) override { return impl_.forward(input); }
    nexus_model::Tensor backward(const nexus_model::Tensor& grad_output) override { return impl_.backward(grad_output); }
    [[nodiscard]] std::vector<nexus_model::Parameter*> parameters() const override { return {}; }
    [[nodiscard]] nxs::LayerSpec nxs_spec() const override { return nxs::spec_of(nxs::LayerKind::Relu, "ReLU"); }

 private:
    mutable nexus_model::ReLU impl_;
};

class Activation : public Module {
 public:
    Activation(std::string name, CustomActivation::ScalarForward forward, CustomActivation::ScalarBackward backward = {})
        : impl_(std::make_shared<CustomActivation>(std::move(name), std::move(forward), std::move(backward))) {}
    void train() override { impl_->train(); }
    void eval() override { impl_->eval(); }
    nexus_model::Tensor forward(const nexus_model::Tensor& input) override { return impl_->forward(input); }
    nexus_model::Tensor backward(const nexus_model::Tensor& grad_output) override { return impl_->backward(grad_output); }
    [[nodiscard]] std::vector<nexus_model::Parameter*> parameters() const override { return {}; }
    [[nodiscard]] nxs::LayerSpec nxs_spec() const override {
        if (impl_->name() == "Swish") {
            return nxs::spec_of(nxs::LayerKind::Swish, "Swish");
        }
        if (impl_->name().empty()) {
            throw std::logic_error("bu katmanin govdesi dosyaya yazilmaz");
        }
        return nxs::spec_of(nxs::LayerKind::NamedActivation, impl_->name());
    }

 private:
    std::shared_ptr<CustomActivation> impl_;
};

inline std::shared_ptr<Module> make_activation(std::string name, CustomActivation::ScalarForward forward,
                                               CustomActivation::ScalarBackward backward = {}) {
    return std::make_shared<Activation>(std::move(name), std::move(forward), std::move(backward));
}

class Dropout : public Module {
 public:
    explicit Dropout(float probability = 0.5f) : probability_(probability), impl_(probability) {}
    void train() override { impl_.train(); }
    void eval() override { impl_.eval(); }
    nexus_model::Tensor forward(const nexus_model::Tensor& input) override { return impl_.forward(input); }
    nexus_model::Tensor backward(const nexus_model::Tensor& grad_output) override { return impl_.backward(grad_output); }
    [[nodiscard]] std::vector<nexus_model::Parameter*> parameters() const override { return {}; }
    [[nodiscard]] nxs::LayerSpec nxs_spec() const override {
        return nxs::spec_of(nxs::LayerKind::Dropout, "Dropout", {nxs::f32("p", probability_)});
    }

 private:
    float probability_ = 0.5f;
    mutable nexus_model::Dropout impl_;
};

class MaxPool2d : public Module {
 public:
    MaxPool2d(int kernel, int stride = -1, int padding = 0)
        : kernel_(kernel), stride_(stride), padding_(padding), impl_(kernel, stride, padding) {}
    nexus_model::Tensor forward(const nexus_model::Tensor& input) override { return impl_.forward(input); }
    nexus_model::Tensor backward(const nexus_model::Tensor& grad_output) override { return impl_.backward(grad_output); }
    [[nodiscard]] std::vector<nexus_model::Parameter*> parameters() const override { return {}; }
    [[nodiscard]] nxs::LayerSpec nxs_spec() const override {
        return nxs::spec_of(nxs::LayerKind::MaxPool2d, "MaxPool2d",
                            {nxs::i32("kernel", kernel_), nxs::i32("stride", stride_), nxs::i32("padding", padding_)});
    }

 private:
    int kernel_ = 1;
    int stride_ = -1;
    int padding_ = 0;
    mutable nexus_model::MaxPool2D impl_;
};

class AvgPool2d : public Module {
 public:
    AvgPool2d(int kernel, int stride = -1, int padding = 0)
        : kernel_(kernel), stride_(stride), padding_(padding), impl_(kernel, stride, padding) {}
    nexus_model::Tensor forward(const nexus_model::Tensor& input) override { return impl_.forward(input); }
    nexus_model::Tensor backward(const nexus_model::Tensor& grad_output) override { return impl_.backward(grad_output); }
    [[nodiscard]] std::vector<nexus_model::Parameter*> parameters() const override { return {}; }
    [[nodiscard]] nxs::LayerSpec nxs_spec() const override {
        return nxs::spec_of(nxs::LayerKind::AvgPool2d, "AvgPool2d",
                            {nxs::i32("kernel", kernel_), nxs::i32("stride", stride_), nxs::i32("padding", padding_)});
    }

 private:
    int kernel_ = 1;
    int stride_ = -1;
    int padding_ = 0;
    mutable nexus_model::AvgPool2D impl_;
};

class GlobalAvgPool2d : public Module {
 public:
    nexus_model::Tensor forward(const nexus_model::Tensor& input) override { return impl_.forward(input); }
    nexus_model::Tensor backward(const nexus_model::Tensor& grad_output) override { return impl_.backward(grad_output); }
    [[nodiscard]] std::vector<nexus_model::Parameter*> parameters() const override { return {}; }
    [[nodiscard]] nxs::LayerSpec nxs_spec() const override { return nxs::spec_of(nxs::LayerKind::GlobalAvgPool2d, "GlobalAvgPool2d"); }

 private:
    mutable nexus_model::GlobalAvgPool2D impl_;
};

class Flatten : public Module {
 public:
    nexus_model::Tensor forward(const nexus_model::Tensor& input) override { return impl_.forward(input); }
    nexus_model::Tensor backward(const nexus_model::Tensor& grad_output) override { return impl_.backward(grad_output); }
    [[nodiscard]] std::vector<nexus_model::Parameter*> parameters() const override { return {}; }
    [[nodiscard]] nxs::LayerSpec nxs_spec() const override { return nxs::spec_of(nxs::LayerKind::Flatten, "Flatten"); }

 private:
    mutable nexus_model::Flatten impl_;
};

class BatchNorm2d : public Module {
 public:
    explicit BatchNorm2d(int num_features) : features_(num_features), impl_(num_features) {}
    void train() override { impl_.train(); }
    void eval() override { impl_.eval(); }
    nexus_model::Tensor forward(const nexus_model::Tensor& input) override { return impl_.forward(input); }
    nexus_model::Tensor backward(const nexus_model::Tensor& grad_output) override { return impl_.backward(grad_output); }
    [[nodiscard]] std::vector<nexus_model::Parameter*> parameters() const override { return impl_.parameters(); }
    void zero_grad() override { impl_.zero_grad(); }
    [[nodiscard]] nxs::LayerSpec nxs_spec() const override {
        return nxs::spec_of(nxs::LayerKind::BatchNorm2d, "BatchNorm2d", {nxs::i32("features", features_)});
    }

 private:
    int features_ = 0;
    mutable nexus_model::BatchNorm2D impl_;
};

class Conv1d : public Module {
 public:
    Conv1d(int in_channels, int out_channels, int kernel, int stride = 1, int padding = 0, bool bias = true)
        : in_(in_channels), out_(out_channels), kernel_(kernel), stride_(stride), padding_(padding), bias_(bias),
          impl_(in_channels, out_channels, kernel, stride, padding, 1, 1, bias) {}
    void train() override { impl_.train(); }
    void eval() override { impl_.eval(); }
    nexus_model::Tensor forward(const nexus_model::Tensor& input) override { return impl_.forward(input); }
    nexus_model::Tensor backward(const nexus_model::Tensor& grad_output) override { return impl_.backward(grad_output); }
    [[nodiscard]] std::vector<nexus_model::Parameter*> parameters() const override { return impl_.parameters(); }
    void zero_grad() override { impl_.zero_grad(); }
    [[nodiscard]] nxs::LayerSpec nxs_spec() const override {
        return nxs::spec_of(nxs::LayerKind::Conv1d, "Conv1d",
                            {nxs::i32("in", in_), nxs::i32("out", out_), nxs::i32("kernel", kernel_), nxs::i32("stride", stride_),
                             nxs::i32("padding", padding_), nxs::boolean("bias", bias_)});
    }

 private:
    int in_ = 0;
    int out_ = 0;
    int kernel_ = 1;
    int stride_ = 1;
    int padding_ = 0;
    bool bias_ = true;
    mutable nexus_model::Conv1D impl_;
};

class Conv3d : public Module {
 public:
    Conv3d(int in_channels, int out_channels, int kernel, int stride = 1, int padding = 0, bool bias = true)
        : in_(in_channels), out_(out_channels), kernel_(kernel), stride_(stride), padding_(padding), bias_(bias),
          impl_(in_channels, out_channels, kernel, stride, padding, 1, 1, bias) {}
    void train() override { impl_.train(); }
    void eval() override { impl_.eval(); }
    nexus_model::Tensor forward(const nexus_model::Tensor& input) override { return impl_.forward(input); }
    nexus_model::Tensor backward(const nexus_model::Tensor& grad_output) override { return impl_.backward(grad_output); }
    [[nodiscard]] std::vector<nexus_model::Parameter*> parameters() const override { return impl_.parameters(); }
    void zero_grad() override { impl_.zero_grad(); }
    [[nodiscard]] nxs::LayerSpec nxs_spec() const override {
        return nxs::spec_of(nxs::LayerKind::Conv3d, "Conv3d",
                            {nxs::i32("in", in_), nxs::i32("out", out_), nxs::i32("kernel", kernel_), nxs::i32("stride", stride_),
                             nxs::i32("padding", padding_), nxs::boolean("bias", bias_)});
    }

 private:
    int in_ = 0;
    int out_ = 0;
    int kernel_ = 1;
    int stride_ = 1;
    int padding_ = 0;
    bool bias_ = true;
    mutable nexus_model::Conv3D impl_;
};

class DepthwiseConv2d : public Module {
 public:
    DepthwiseConv2d(int channels, int kernel, int stride = 1, int padding = 0)
        : channels_(channels), kernel_(kernel), stride_(stride), padding_(padding), impl_(channels, kernel, stride, padding) {}
    void train() override { impl_.train(); }
    void eval() override { impl_.eval(); }
    nexus_model::Tensor forward(const nexus_model::Tensor& input) override { return impl_.forward(input); }
    nexus_model::Tensor backward(const nexus_model::Tensor& grad_output) override { return impl_.backward(grad_output); }
    [[nodiscard]] std::vector<nexus_model::Parameter*> parameters() const override { return impl_.parameters(); }
    void zero_grad() override { impl_.zero_grad(); }
    [[nodiscard]] nxs::LayerSpec nxs_spec() const override {
        return nxs::spec_of(nxs::LayerKind::DepthwiseConv2d, "DepthwiseConv2d",
                            {nxs::i32("channels", channels_), nxs::i32("kernel", kernel_), nxs::i32("stride", stride_),
                             nxs::i32("padding", padding_)});
    }

 private:
    int channels_ = 0;
    int kernel_ = 1;
    int stride_ = 1;
    int padding_ = 0;
    mutable nexus_model::DepthwiseConv2D impl_;
};

class GroupNorm : public Module {
 public:
    GroupNorm(int num_groups, int num_channels) : groups_(num_groups), channels_(num_channels), impl_(num_groups, num_channels) {}
    void train() override { impl_.train(); }
    void eval() override { impl_.eval(); }
    nexus_model::Tensor forward(const nexus_model::Tensor& input) override { return impl_.forward(input); }
    nexus_model::Tensor backward(const nexus_model::Tensor& grad_output) override { return impl_.backward(grad_output); }
    [[nodiscard]] std::vector<nexus_model::Parameter*> parameters() const override { return impl_.parameters(); }
    void zero_grad() override { impl_.zero_grad(); }
    [[nodiscard]] nxs::LayerSpec nxs_spec() const override {
        return nxs::spec_of(nxs::LayerKind::GroupNorm, "GroupNorm", {nxs::i32("groups", groups_), nxs::i32("channels", channels_)});
    }

 private:
    int groups_ = 1;
    int channels_ = 0;
    mutable nexus_model::GroupNorm impl_;
};

class InstanceNorm : public Module {
 public:
    explicit InstanceNorm(int num_channels) : channels_(num_channels), impl_(num_channels) {}
    void train() override { impl_.train(); }
    void eval() override { impl_.eval(); }
    nexus_model::Tensor forward(const nexus_model::Tensor& input) override { return impl_.forward(input); }
    nexus_model::Tensor backward(const nexus_model::Tensor& grad_output) override { return impl_.backward(grad_output); }
    [[nodiscard]] std::vector<nexus_model::Parameter*> parameters() const override { return impl_.parameters(); }
    void zero_grad() override { impl_.zero_grad(); }
    [[nodiscard]] nxs::LayerSpec nxs_spec() const override {
        return nxs::spec_of(nxs::LayerKind::InstanceNorm, "InstanceNorm", {nxs::i32("channels", channels_)});
    }

 private:
    int channels_ = 0;
    mutable nexus_model::InstanceNorm impl_;
};

class Residual : public Module {
 public:
    explicit Residual(std::shared_ptr<Module> function) : function_(std::move(function)) {
        if (!function_) {
            throw nexus_model::ModelError("Residual received an empty module");
        }
    }

    void train() override { function_->train(); }
    void eval() override { function_->eval(); }

    nexus_model::Tensor forward(const nexus_model::Tensor& input) override {
        saved_.copy_from(input);
        nexus_model::Tensor function_value = function_->forward(saved_);
        if (function_value.numel() != saved_.numel()) {
            throw nexus_model::ModelError("Residual function must keep the element count");
        }
        output_.resize_like(function_value);
        nexus_model::kernels::add(function_value.data(), saved_.data(), output_.data(), saved_.numel());
        return output_;
    }

    nexus_model::Tensor backward(const nexus_model::Tensor& grad_output) override {
        nexus_model::Tensor function_grad = function_->backward(grad_output);
        grad_input_.resize_like(grad_output);
        nexus_model::kernels::add(function_grad.data(), grad_output.data(), grad_input_.data(), grad_output.numel());
        return grad_input_;
    }

    [[nodiscard]] std::vector<nexus_model::Parameter*> parameters() const override { return function_->parameters(); }

    void zero_grad() override { function_->zero_grad(); }

    [[nodiscard]] nxs::LayerSpec nxs_spec() const override {
        nxs::LayerSpec spec = nxs::spec_of(nxs::LayerKind::Residual, "Residual");
        spec.children.push_back(function_->nxs_spec());
        return spec;
    }

 private:
    std::shared_ptr<Module> function_;
    nexus_model::Tensor saved_;
    nexus_model::Tensor output_;
    nexus_model::Tensor grad_input_;
};

class Reshape : public Module {
 public:
    explicit Reshape(std::initializer_list<std::size_t> shape) : shape_(shape), impl_(shape_) {}
    explicit Reshape(std::vector<std::size_t> shape) : shape_(std::move(shape)), impl_(shape_) {}
    nexus_model::Tensor forward(const nexus_model::Tensor& input) override { return impl_.forward(input); }
    nexus_model::Tensor backward(const nexus_model::Tensor& grad_output) override { return impl_.backward(grad_output); }
    [[nodiscard]] std::vector<nexus_model::Parameter*> parameters() const override { return {}; }
    [[nodiscard]] nxs::LayerSpec nxs_spec() const override {
        std::vector<nxs::Attr> attrs;
        for (std::size_t dim : shape_) {
            attrs.push_back(nxs::i32("dim", static_cast<int>(dim)));
        }
        return nxs::spec_of(nxs::LayerKind::Reshape, "Reshape", std::move(attrs));
    }

 private:
    std::vector<std::size_t> shape_;
    mutable nexus_model::Reshape impl_;
};

class RNN : public Module {
 public:
    RNN(int input_size, int hidden_size) : input_(input_size), hidden_(hidden_size), impl_(input_size, hidden_size) {}
    void train() override { impl_.train(); }
    void eval() override { impl_.eval(); }
    nexus_model::Tensor forward(const nexus_model::Tensor& input) override { return impl_.forward(input); }
    nexus_model::Tensor backward(const nexus_model::Tensor& grad_output) override { return impl_.backward(grad_output); }
    [[nodiscard]] std::vector<nexus_model::Parameter*> parameters() const override { return impl_.parameters(); }
    void zero_grad() override { impl_.zero_grad(); }
    [[nodiscard]] nxs::LayerSpec nxs_spec() const override {
        return nxs::spec_of(nxs::LayerKind::Rnn, "RNN", {nxs::i32("input", input_), nxs::i32("hidden", hidden_)});
    }

 private:
    int input_ = 0;
    int hidden_ = 0;
    mutable nexus_model::RNN impl_;
};

class PositionalEncoding : public Module {
 public:
    PositionalEncoding(int embed, int max_len = 512, bool learnable = false)
        : embed_(embed), max_len_(max_len), learnable_(learnable), impl_(embed, max_len, learnable) {}
    void train() override { impl_.train(); }
    void eval() override { impl_.eval(); }
    nexus_model::Tensor forward(const nexus_model::Tensor& input) override { return impl_.forward(input); }
    nexus_model::Tensor backward(const nexus_model::Tensor& grad_output) override { return impl_.backward(grad_output); }
    [[nodiscard]] std::vector<nexus_model::Parameter*> parameters() const override { return impl_.parameters(); }
    void zero_grad() override { impl_.zero_grad(); }
    [[nodiscard]] nxs::LayerSpec nxs_spec() const override {
        return nxs::spec_of(nxs::LayerKind::PositionalEncoding, "PositionalEncoding",
                            {nxs::i32("embed", embed_), nxs::i32("max_len", max_len_), nxs::boolean("learnable", learnable_)});
    }

 private:
    int embed_ = 0;
    int max_len_ = 512;
    bool learnable_ = false;
    mutable nexus_model::PositionalEncoding impl_;
};

class TransformerDecoder : public Module {
 public:
    TransformerDecoder(int embed, int heads, int feedforward = 0)
        : embed_(embed), heads_(heads), feedforward_(feedforward), impl_(embed, heads, feedforward, 0.0f) {}
    void train() override { impl_.train(); }
    void eval() override { impl_.eval(); }
    nexus_model::Tensor forward(const nexus_model::Tensor& target, const nexus_model::Tensor& memory) { return impl_.forward(target, memory); }
    nexus_model::Tensor forward(const nexus_model::Tensor& input) override {
        (void)input;
        throw nexus_model::ModelError("TransformerDecoder requires forward(target, memory)");
    }
    nexus_model::Tensor backward(const nexus_model::Tensor& grad_output) override { return impl_.backward(grad_output); }
    [[nodiscard]] std::vector<nexus_model::Parameter*> parameters() const override { return impl_.parameters(); }
    void zero_grad() override { impl_.zero_grad(); }
    [[nodiscard]] nxs::LayerSpec nxs_spec() const override {
        return nxs::spec_of(nxs::LayerKind::TransformerDecoder, "TransformerDecoder",
                            {nxs::i32("embed", embed_), nxs::i32("heads", heads_), nxs::i32("feedforward", feedforward_)});
    }

 private:
    int embed_ = 0;
    int heads_ = 1;
    int feedforward_ = 0;
    mutable nexus_model::TransformerDecoderLayer impl_;
};

class Sequential : public Module {
 public:
    template <typename Layer, typename... Args>
    Sequential& add(Args&&... args) {
        layers_->push_back(std::make_shared<Layer>(std::forward<Args>(args)...));
        return *this;
    }

    Sequential& add(std::shared_ptr<Module> layer) {
        if (!layer) {
            throw nexus_model::ModelError("Sequential add received an empty module");
        }
        layers_->push_back(std::move(layer));
        return *this;
    }

    nexus_model::Tensor forward(const nexus_model::Tensor& input) override {
        nexus_model::Tensor current = input;
        for (const std::shared_ptr<Module>& layer : *layers_) {
            current = layer->forward(current);
        }
        return current;
    }

    nexus_model::Tensor backward(const nexus_model::Tensor& grad_output) override {
        nexus_model::Tensor grad = grad_output;
        for (auto it = layers_->rbegin(); it != layers_->rend(); ++it) {
            grad = (*it)->backward(grad);
        }
        return grad;
    }

    [[nodiscard]] std::vector<nexus_model::Parameter*> parameters() const override {
        std::vector<nexus_model::Parameter*> values;
        for (const std::shared_ptr<Module>& layer : *layers_) {
            auto owned = layer->parameters();
            values.insert(values.end(), owned.begin(), owned.end());
        }
        return values;
    }

    void zero_grad() override {
        for (const std::shared_ptr<Module>& layer : *layers_) {
            layer->zero_grad();
        }
    }

    void train() override {
        for (const std::shared_ptr<Module>& layer : *layers_) {
            layer->train();
        }
    }

    void eval() override {
        for (const std::shared_ptr<Module>& layer : *layers_) {
            layer->eval();
        }
    }

    [[nodiscard]] std::size_t size() const noexcept { return layers_->size(); }

    [[nodiscard]] nxs::LayerSpec nxs_spec() const override {
        nxs::LayerSpec spec = nxs::spec_of(nxs::LayerKind::Sequential, "Sequential");
        for (const std::shared_ptr<Module>& layer : *layers_) {
            spec.children.push_back(layer->nxs_spec());
        }
        return spec;
    }

    void save_nxs(const std::string& path) const {
        nxs::Archive archive;
        for (const std::shared_ptr<Module>& layer : *layers_) {
            archive.layers.push_back(layer->nxs_spec());
        }
        archive.weights = nxs::collect_weights(parameters());
        nxs::save_archive(path, archive);
    }

    [[nodiscard]] static Sequential load_nxs(const std::string& path) {
        const nxs::Archive archive = nxs::load_archive(path);
        Sequential model;
        for (const nxs::LayerSpec& spec : archive.layers) {
            model.add(nxs::module_from_spec(spec));
        }
        nxs::apply_weights(model.parameters(), archive.weights);
        return model;
    }

    Sequential* operator->() noexcept { return this; }
    const Sequential* operator->() const noexcept { return this; }

 private:
    std::shared_ptr<std::vector<std::shared_ptr<Module>>> layers_ = std::make_shared<std::vector<std::shared_ptr<Module>>>();
};

#include "nexus/nn/graph.inl"

}  // namespace nn
}  // namespace nexus
