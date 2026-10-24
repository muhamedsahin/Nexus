/**
 * @file nxs.cpp
 * @brief NXS2 tek dosya. NXS1 egitmen kaydi model diye acilmaz.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/model/nxs_format.hpp"
#include "nexus/nn/activations.hpp"
#include "nexus/nn/modules.hpp"

#include <cstring>
#include <fstream>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace {
std::map<std::string, std::function<std::shared_ptr<nexus::nn::Module>()>>& activation_factories() {
    // Yorumlayici kapanirken py::function yok edicisi GIL ister. Tablo kasitli sizdir.
    static auto* table = new std::map<std::string, std::function<std::shared_ptr<nexus::nn::Module>()>>();
    return *table;
}

std::map<std::string, std::function<std::shared_ptr<nexus::nn::Module>()>>& module_factories() {
    static auto* table = new std::map<std::string, std::function<std::shared_ptr<nexus::nn::Module>()>>();
    return *table;
}

std::vector<std::uint8_t>& pending_python() {
    static std::vector<std::uint8_t> bytes;
    return bytes;
}
}  // namespace

namespace nexus {
namespace nxs {
namespace {

constexpr std::uint16_t kFlagOptim = 1;
constexpr std::uint16_t kFlagTrainer = 2;
constexpr std::uint16_t kFlagRl = 4;
constexpr std::uint16_t kFlagPython = 8;

void fail(const std::string& message) { throw std::runtime_error(message); }

void write_u8(std::ostream& out, std::uint8_t value) { out.write(reinterpret_cast<const char*>(&value), 1); }
void write_u16(std::ostream& out, std::uint16_t value) { out.write(reinterpret_cast<const char*>(&value), sizeof(value)); }
void write_u32(std::ostream& out, std::uint32_t value) { out.write(reinterpret_cast<const char*>(&value), sizeof(value)); }
void write_u64(std::ostream& out, std::uint64_t value) { out.write(reinterpret_cast<const char*>(&value), sizeof(value)); }
void write_f32(std::ostream& out, float value) { out.write(reinterpret_cast<const char*>(&value), sizeof(value)); }
void write_f64(std::ostream& out, double value) { out.write(reinterpret_cast<const char*>(&value), sizeof(value)); }

std::uint8_t read_u8(std::istream& in) {
    std::uint8_t value = 0;
    in.read(reinterpret_cast<char*>(&value), 1);
    return value;
}
std::uint16_t read_u16(std::istream& in) {
    std::uint16_t value = 0;
    in.read(reinterpret_cast<char*>(&value), sizeof(value));
    return value;
}
std::uint32_t read_u32(std::istream& in) {
    std::uint32_t value = 0;
    in.read(reinterpret_cast<char*>(&value), sizeof(value));
    return value;
}
std::uint64_t read_u64(std::istream& in) {
    std::uint64_t value = 0;
    in.read(reinterpret_cast<char*>(&value), sizeof(value));
    return value;
}
float read_f32(std::istream& in) {
    float value = 0.0f;
    in.read(reinterpret_cast<char*>(&value), sizeof(value));
    return value;
}
double read_f64(std::istream& in) {
    double value = 0.0;
    in.read(reinterpret_cast<char*>(&value), sizeof(value));
    return value;
}

void write_bytes(std::ostream& out, const std::string& bytes) {
    if (bytes.size() > static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
        fail("nxs name is too long");
    }
    write_u32(out, static_cast<std::uint32_t>(bytes.size()));
    if (!bytes.empty()) {
        out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    }
}

std::string read_bytes(std::istream& in) {
    const std::uint32_t n = read_u32(in);
    std::string bytes(n, '\0');
    if (n > 0) {
        in.read(bytes.data(), static_cast<std::streamsize>(n));
    }
    return bytes;
}

void write_attr(std::ostream& out, const Attr& attr) {
    write_u8(out, static_cast<std::uint8_t>(attr.kind));
    write_bytes(out, attr.key);
    if (attr.kind == AttrKind::I32) {
        write_u32(out, static_cast<std::uint32_t>(attr.i));
    } else if (attr.kind == AttrKind::F32) {
        write_f32(out, attr.f);
    } else if (attr.kind == AttrKind::Bool) {
        write_u8(out, attr.b ? 1 : 0);
    } else if (attr.kind == AttrKind::String) {
        write_bytes(out, attr.s);
    } else {
        fail("nxs attribute kind is invalid");
    }
}

Attr read_attr(std::istream& in) {
    Attr attr;
    attr.kind = static_cast<AttrKind>(read_u8(in));
    attr.key = read_bytes(in);
    if (attr.kind == AttrKind::I32) {
        attr.i = static_cast<std::int32_t>(read_u32(in));
    } else if (attr.kind == AttrKind::F32) {
        attr.f = read_f32(in);
    } else if (attr.kind == AttrKind::Bool) {
        attr.b = read_u8(in) != 0;
    } else if (attr.kind == AttrKind::String) {
        attr.s = read_bytes(in);
    } else {
        fail("nxs attribute kind is invalid");
    }
    return attr;
}

void write_layer(std::ostream& out, const LayerSpec& spec) {
    write_u16(out, spec.type);
    write_bytes(out, spec.name);
    write_u32(out, static_cast<std::uint32_t>(spec.attrs.size()));
    for (const Attr& attr : spec.attrs) {
        write_attr(out, attr);
    }
    write_u32(out, static_cast<std::uint32_t>(spec.children.size()));
    for (const LayerSpec& child : spec.children) {
        write_layer(out, child);
    }
}

LayerSpec read_layer(std::istream& in) {
    LayerSpec spec;
    spec.type = read_u16(in);
    spec.name = read_bytes(in);
    const std::uint32_t n_attr = read_u32(in);
    spec.attrs.resize(n_attr);
    for (std::uint32_t i = 0; i < n_attr; ++i) {
        spec.attrs[i] = read_attr(in);
    }
    const std::uint32_t n_child = read_u32(in);
    spec.children.resize(n_child);
    for (std::uint32_t i = 0; i < n_child; ++i) {
        spec.children[i] = read_layer(in);
    }
    return spec;
}

void write_weights(std::ostream& out, const std::vector<Weight>& weights) {
    write_u32(out, static_cast<std::uint32_t>(weights.size()));
    for (const Weight& weight : weights) {
        write_bytes(out, weight.name);
        write_u32(out, static_cast<std::uint32_t>(weight.dims.size()));
        for (std::uint64_t dim : weight.dims) {
            write_u64(out, dim);
        }
        write_u64(out, static_cast<std::uint64_t>(weight.values.size()));
        if (!weight.values.empty()) {
            out.write(reinterpret_cast<const char*>(weight.values.data()),
                      static_cast<std::streamsize>(weight.values.size() * sizeof(float)));
        }
    }
}

std::vector<Weight> read_weights(std::istream& in) {
    const std::uint32_t count = read_u32(in);
    std::vector<Weight> weights(count);
    for (std::uint32_t i = 0; i < count; ++i) {
        Weight& weight = weights[i];
        weight.name = read_bytes(in);
        const std::uint32_t rank = read_u32(in);
        weight.dims.resize(rank);
        for (std::uint32_t d = 0; d < rank; ++d) {
            weight.dims[d] = read_u64(in);
        }
        const auto n = static_cast<std::size_t>(read_u64(in));
        weight.values.resize(n);
        if (n > 0) {
            in.read(reinterpret_cast<char*>(weight.values.data()), static_cast<std::streamsize>(n * sizeof(float)));
        }
    }
    return weights;
}

void write_optim(std::ostream& out, const nexus_optim::OptimizerState& state) {
    out.write("NXO1", 4);
    write_bytes(out, state.algorithm);
    write_u64(out, state.step);
    write_u32(out, static_cast<std::uint32_t>(state.group_learning_rates.size()));
    for (double lr : state.group_learning_rates) {
        write_f64(out, lr);
    }
    write_u32(out, static_cast<std::uint32_t>(state.blobs.size()));
    for (const auto& blob : state.blobs) {
        write_u64(out, static_cast<std::uint64_t>(blob.size()));
        if (!blob.empty()) {
            out.write(reinterpret_cast<const char*>(blob.data()), static_cast<std::streamsize>(blob.size()));
        }
    }
    write_u32(out, static_cast<std::uint32_t>(state.scalars.size()));
    for (double value : state.scalars) {
        write_f64(out, value);
    }
}

nexus_optim::OptimizerState read_optim(std::istream& in) {
    char magic[4] = {};
    in.read(magic, 4);
    if (std::strncmp(magic, "NXO1", 4) != 0) {
        fail("optimizer block magic mismatch");
    }
    nexus_optim::OptimizerState state;
    state.algorithm = read_bytes(in);
    state.step = read_u64(in);
    const std::uint32_t n_lr = read_u32(in);
    state.group_learning_rates.resize(n_lr);
    for (std::uint32_t i = 0; i < n_lr; ++i) {
        state.group_learning_rates[i] = read_f64(in);
    }
    const std::uint32_t n_blob = read_u32(in);
    state.blobs.resize(n_blob);
    for (std::uint32_t i = 0; i < n_blob; ++i) {
        const auto n = static_cast<std::size_t>(read_u64(in));
        state.blobs[i].resize(n);
        if (n > 0) {
            in.read(reinterpret_cast<char*>(state.blobs[i].data()), static_cast<std::streamsize>(n));
        }
    }
    const std::uint32_t n_scalar = read_u32(in);
    state.scalars.resize(n_scalar);
    for (std::uint32_t i = 0; i < n_scalar; ++i) {
        state.scalars[i] = read_f64(in);
    }
    return state;
}

void write_trainer(std::ostream& out, const TrainerRecord& state) {
    out.write("NXS1", 4);
    write_u32(out, static_cast<std::uint32_t>(state.epoch));
    write_u32(out, static_cast<std::uint32_t>(state.global_step));
    write_f64(out, state.current_loss);
    write_f64(out, state.best_metric);
    write_u8(out, state.should_stop ? 1 : 0);
    write_u32(out, static_cast<std::uint32_t>(state.metrics.size()));
    for (const auto& [name, value] : state.metrics) {
        write_bytes(out, name);
        write_f64(out, value);
    }
}

TrainerRecord read_trainer(std::istream& in) {
    char magic[4] = {};
    in.read(magic, 4);
    if (std::strncmp(magic, "NXS1", 4) != 0) {
        fail("trainer block magic mismatch");
    }
    TrainerRecord state;
    state.epoch = static_cast<std::int32_t>(read_u32(in));
    state.global_step = static_cast<std::int32_t>(read_u32(in));
    state.current_loss = read_f64(in);
    state.best_metric = read_f64(in);
    state.should_stop = read_u8(in) != 0;
    const std::uint32_t n = read_u32(in);
    state.metrics.reserve(n);
    for (std::uint32_t i = 0; i < n; ++i) {
        const std::string name = read_bytes(in);
        state.metrics.emplace_back(name, read_f64(in));
    }
    return state;
}

void write_floats(std::ostream& out, const std::vector<float>& values) {
    write_u64(out, static_cast<std::uint64_t>(values.size()));
    if (!values.empty()) {
        out.write(reinterpret_cast<const char*>(values.data()), static_cast<std::streamsize>(values.size() * sizeof(float)));
    }
}

std::vector<float> read_floats(std::istream& in) {
    const auto n = static_cast<std::size_t>(read_u64(in));
    std::vector<float> values(n);
    if (n > 0) {
        in.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(n * sizeof(float)));
    }
    return values;
}

void write_rl(std::ostream& out, const RlRecord& rl) {
    write_bytes(out, rl.kind);
    write_u32(out, static_cast<std::uint32_t>(rl.config.size()));
    for (const Attr& attr : rl.config) {
        write_attr(out, attr);
    }
    write_u32(out, static_cast<std::uint32_t>(rl.nets.size()));
    for (const MlpBlob& net : rl.nets) {
        write_bytes(out, net.name);
        write_u32(out, static_cast<std::uint32_t>(net.sizes.size()));
        for (std::int32_t size : net.sizes) {
            write_u32(out, static_cast<std::uint32_t>(size));
        }
        write_u64(out, net.adam_step);
        write_floats(out, net.parameters);
        write_floats(out, net.moments);
    }
    write_u32(out, static_cast<std::uint32_t>(rl.elos.size()));
    for (float elo : rl.elos) {
        write_f32(out, elo);
    }
    write_u32(out, static_cast<std::uint32_t>(rl.player_weights.size()));
    for (const auto& weights : rl.player_weights) {
        write_floats(out, weights);
    }
}

RlRecord read_rl(std::istream& in) {
    RlRecord rl;
    rl.kind = read_bytes(in);
    const std::uint32_t n_attr = read_u32(in);
    rl.config.resize(n_attr);
    for (std::uint32_t i = 0; i < n_attr; ++i) {
        rl.config[i] = read_attr(in);
    }
    const std::uint32_t n_net = read_u32(in);
    rl.nets.resize(n_net);
    for (std::uint32_t i = 0; i < n_net; ++i) {
        MlpBlob& net = rl.nets[i];
        net.name = read_bytes(in);
        const std::uint32_t n_size = read_u32(in);
        net.sizes.resize(n_size);
        for (std::uint32_t s = 0; s < n_size; ++s) {
            net.sizes[s] = static_cast<std::int32_t>(read_u32(in));
        }
        net.adam_step = read_u64(in);
        net.parameters = read_floats(in);
        net.moments = read_floats(in);
    }
    const std::uint32_t n_elo = read_u32(in);
    rl.elos.resize(n_elo);
    for (std::uint32_t i = 0; i < n_elo; ++i) {
        rl.elos[i] = read_f32(in);
    }
    const std::uint32_t n_player = read_u32(in);
    rl.player_weights.resize(n_player);
    for (std::uint32_t i = 0; i < n_player; ++i) {
        rl.player_weights[i] = read_floats(in);
    }
    return rl;
}

std::shared_ptr<nn::Module> make_one(const LayerSpec& spec);

std::string require_string(const LayerSpec& spec, const std::string& key) {
    const Attr* attr = find_attr(spec, key);
    if (attr == nullptr || attr->kind != AttrKind::String) {
        fail("nxs attribute missing: " + key);
    }
    return attr->s;
}

std::vector<std::string> split_names(const std::string& joined) {
    std::vector<std::string> names;
    if (joined.empty()) {
        return names;
    }
    std::stringstream stream(joined);
    std::string item;
    while (std::getline(stream, item, ',')) {
        names.push_back(item);
    }
    return names;
}

std::shared_ptr<nn::Graph> make_graph(const LayerSpec& spec) {
    auto graph = std::make_shared<nn::Graph>();
    for (const LayerSpec& child : spec.children) {
        graph->restore(require_string(child, "graph_name"), make_one(child), split_names(require_string(child, "graph_inputs")),
                       require_i32(child, "graph_merge"));
    }
    return graph;
}

std::shared_ptr<nn::Sequential> make_sequential(const LayerSpec& spec) {
    auto model = std::make_shared<nn::Sequential>();
    for (const LayerSpec& child : spec.children) {
        model->add(make_one(child));
    }
    return model;
}

std::shared_ptr<nn::Module> make_one(const LayerSpec& spec) {
    const auto kind = static_cast<LayerKind>(spec.type);
    switch (kind) {
        case LayerKind::Linear:
            return std::make_shared<nn::Linear>(static_cast<std::size_t>(require_i32(spec, "in")),
                                                static_cast<std::size_t>(require_i32(spec, "out")), require_bool(spec, "bias"));
        case LayerKind::Conv2d:
            return std::make_shared<nn::Conv2d>(require_i32(spec, "in"), require_i32(spec, "out"), require_i32(spec, "kernel"),
                                                require_i32(spec, "stride"), require_i32(spec, "padding"), require_bool(spec, "bias"));
        case LayerKind::MultiHeadAttention:
            return std::make_shared<nn::MultiHeadAttention>(require_i32(spec, "embed"), require_i32(spec, "heads"));
        case LayerKind::Lstm:
            return std::make_shared<nn::LSTM>(require_i32(spec, "input"), require_i32(spec, "hidden"));
        case LayerKind::Gru:
            return std::make_shared<nn::GRU>(require_i32(spec, "input"), require_i32(spec, "hidden"));
        case LayerKind::TransformerEncoder:
            return std::make_shared<nn::TransformerEncoder>(require_i32(spec, "embed"), require_i32(spec, "heads"),
                                                           require_i32(spec, "feedforward"));
        case LayerKind::Embedding:
            return std::make_shared<nn::Embedding>(require_i32(spec, "num"), require_i32(spec, "dim"), require_i32(spec, "padding"));
        case LayerKind::BatchNorm1d:
            return std::make_shared<nn::BatchNorm1d>(require_i32(spec, "features"));
        case LayerKind::LayerNorm:
            return std::make_shared<nn::LayerNorm>(static_cast<std::size_t>(require_i32(spec, "size")), require_f32(spec, "eps"));
        case LayerKind::Relu:
            return std::make_shared<nn::ReLU>();
        case LayerKind::Swish:
            return nn::swish();
        case LayerKind::NamedActivation: {
            const auto found = activation_factories().find(spec.name);
            if (found == activation_factories().end() || !found->second) {
                fail("python govdesi bu surecte yok: " + spec.name);
            }
            return found->second();
        }
        case LayerKind::PythonModule: {
            const auto found = module_factories().find(spec.name);
            if (found == module_factories().end() || !found->second) {
                fail("python govdesi bu surecte yok: " + spec.name);
            }
            return found->second();
        }
        case LayerKind::Dropout:
            return std::make_shared<nn::Dropout>(require_f32(spec, "p"));
        case LayerKind::MaxPool2d:
            return std::make_shared<nn::MaxPool2d>(require_i32(spec, "kernel"), require_i32(spec, "stride"), require_i32(spec, "padding"));
        case LayerKind::AvgPool2d:
            return std::make_shared<nn::AvgPool2d>(require_i32(spec, "kernel"), require_i32(spec, "stride"), require_i32(spec, "padding"));
        case LayerKind::GlobalAvgPool2d:
            return std::make_shared<nn::GlobalAvgPool2d>();
        case LayerKind::Flatten:
            return std::make_shared<nn::Flatten>();
        case LayerKind::BatchNorm2d:
            return std::make_shared<nn::BatchNorm2d>(require_i32(spec, "features"));
        case LayerKind::Conv1d:
            return std::make_shared<nn::Conv1d>(require_i32(spec, "in"), require_i32(spec, "out"), require_i32(spec, "kernel"),
                                                require_i32(spec, "stride"), require_i32(spec, "padding"), require_bool(spec, "bias"));
        case LayerKind::Conv3d:
            return std::make_shared<nn::Conv3d>(require_i32(spec, "in"), require_i32(spec, "out"), require_i32(spec, "kernel"),
                                                require_i32(spec, "stride"), require_i32(spec, "padding"), require_bool(spec, "bias"));
        case LayerKind::DepthwiseConv2d:
            return std::make_shared<nn::DepthwiseConv2d>(require_i32(spec, "channels"), require_i32(spec, "kernel"),
                                                        require_i32(spec, "stride"), require_i32(spec, "padding"));
        case LayerKind::GroupNorm:
            return std::make_shared<nn::GroupNorm>(require_i32(spec, "groups"), require_i32(spec, "channels"));
        case LayerKind::InstanceNorm:
            return std::make_shared<nn::InstanceNorm>(require_i32(spec, "channels"));
        case LayerKind::Residual: {
            if (spec.children.size() != 1) {
                fail("residual nxs node needs one child");
            }
            return std::make_shared<nn::Residual>(make_one(spec.children[0]));
        }
        case LayerKind::Reshape: {
            std::vector<std::size_t> shape;
            for (const Attr& attr : spec.attrs) {
                if (attr.kind == AttrKind::I32) {
                    shape.push_back(static_cast<std::size_t>(attr.i));
                }
            }
            return std::make_shared<nn::Reshape>(shape);
        }
        case LayerKind::Rnn:
            return std::make_shared<nn::RNN>(require_i32(spec, "input"), require_i32(spec, "hidden"));
        case LayerKind::PositionalEncoding:
            return std::make_shared<nn::PositionalEncoding>(require_i32(spec, "embed"), require_i32(spec, "max_len"),
                                                           require_bool(spec, "learnable"));
        case LayerKind::TransformerDecoder:
            return std::make_shared<nn::TransformerDecoder>(require_i32(spec, "embed"), require_i32(spec, "heads"),
                                                           require_i32(spec, "feedforward"));
        case LayerKind::Sequential:
            return make_sequential(spec);
        case LayerKind::Graph:
            return make_graph(spec);
        case LayerKind::Concat:
            fail("Concat is a graph merge, not a standalone layer");
        case LayerKind::CausalBlock:
            return std::make_shared<nn::CausalTransformerBlock>(require_i32(spec, "embed"), require_i32(spec, "heads"),
                                                               require_i32(spec, "feedforward"));
        case LayerKind::DecoderBlock:
        case LayerKind::LanguageModel:
            fail("DecoderBlock/LanguageModel autograd tabanlidir: nexus::lm::LanguageModel::load_nxs ile yukleyin");
        default:
            fail("nxs layer type is unknown");
    }
    return nullptr;
}

}  // namespace

void set_python_payload(std::vector<std::uint8_t> bytes) { pending_python() = std::move(bytes); }

void register_named_activation(std::string name, std::function<std::shared_ptr<nn::Module>()> factory) {
    activation_factories()[std::move(name)] = std::move(factory);
}

void register_named_module(std::string name, std::function<std::shared_ptr<nn::Module>()> factory) {
    module_factories()[std::move(name)] = std::move(factory);
}

void save_archive(const std::string& path, const Archive& archive) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        fail("cannot open nxs file for writing: " + path);
    }
    out.write("NXS2", 4);
    write_u16(out, 1);
    std::uint16_t flags = 0;
    if (archive.has_optim) {
        flags = static_cast<std::uint16_t>(flags | kFlagOptim);
    }
    if (archive.has_trainer) {
        flags = static_cast<std::uint16_t>(flags | kFlagTrainer);
    }
    if (archive.has_rl) {
        flags = static_cast<std::uint16_t>(flags | kFlagRl);
    }
    std::vector<std::uint8_t> python_blob = std::move(pending_python());
    pending_python().clear();
    if (python_blob.empty()) {
        python_blob = archive.python_bytes;
    }
    const bool has_python = !python_blob.empty();
    if (has_python) {
        flags = static_cast<std::uint16_t>(flags | kFlagPython);
    }
    write_u16(out, flags);
    write_u32(out, static_cast<std::uint32_t>(archive.layers.size()));
    for (const LayerSpec& layer : archive.layers) {
        write_layer(out, layer);
    }
    write_weights(out, archive.weights);
    if (archive.has_optim) {
        write_optim(out, archive.optim);
    }
    if (archive.has_trainer) {
        write_trainer(out, archive.trainer);
    }
    if (archive.has_rl) {
        write_rl(out, archive.rl);
    }
    if (has_python) {
        write_u64(out, static_cast<std::uint64_t>(python_blob.size()));
        out.write(reinterpret_cast<const char*>(python_blob.data()), static_cast<std::streamsize>(python_blob.size()));
    }
    if (!out) {
        fail("failed while writing nxs file");
    }
}

Archive load_archive(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        fail("cannot open nxs file for reading: " + path);
    }
    char magic[4] = {};
    in.read(magic, 4);
    if (std::strncmp(magic, "NXS1", 4) == 0) {
        fail("bu egitmen durumu, model arsivi degil");
    }
    if (std::strncmp(magic, "NXS2", 4) != 0) {
        fail("nxs magic mismatch");
    }
    const std::uint16_t version = read_u16(in);
    if (version != 1) {
        fail("nxs version is unsupported");
    }
    const std::uint16_t flags = read_u16(in);
    Archive archive;
    const std::uint32_t n_layer = read_u32(in);
    archive.layers.resize(n_layer);
    for (std::uint32_t i = 0; i < n_layer; ++i) {
        archive.layers[i] = read_layer(in);
    }
    archive.weights = read_weights(in);
    if ((flags & kFlagOptim) != 0) {
        archive.has_optim = true;
        archive.optim = read_optim(in);
    }
    if ((flags & kFlagTrainer) != 0) {
        archive.has_trainer = true;
        archive.trainer = read_trainer(in);
    }
    if ((flags & kFlagRl) != 0) {
        archive.has_rl = true;
        archive.rl = read_rl(in);
    }
    if ((flags & kFlagPython) != 0) {
        const std::uint64_t n = read_u64(in);
        if (n > 32ull * 1024ull * 1024ull) {
            fail("nxs python payload is too large");
        }
        archive.python_bytes.resize(static_cast<std::size_t>(n));
        if (n != 0) {
            in.read(reinterpret_cast<char*>(archive.python_bytes.data()), static_cast<std::streamsize>(n));
        }
    }
    if (!in) {
        fail("truncated nxs file");
    }
    return archive;
}

std::vector<Weight> collect_weights(const std::vector<nexus_model::Parameter*>& parameters) {
    std::vector<Weight> weights;
    weights.reserve(parameters.size());
    for (std::size_t index = 0; index < parameters.size(); ++index) {
        const nexus_model::Parameter* parameter = parameters[index];
        Weight weight;
        weight.name = std::to_string(index);
        const std::vector<std::size_t> shape = parameter->data.shape_vec();
        weight.dims.assign(shape.begin(), shape.end());
        weight.values.assign(parameter->data.data(), parameter->data.data() + parameter->data.numel());
        weights.push_back(std::move(weight));
    }
    return weights;
}

void apply_weights(const std::vector<nexus_model::Parameter*>& parameters, const std::vector<Weight>& weights) {
    if (parameters.size() != weights.size()) {
        fail("nxs weight count does not match the model");
    }
    for (std::size_t index = 0; index < parameters.size(); ++index) {
        nexus_model::Parameter* parameter = parameters[index];
        if (parameter->data.numel() != weights[index].values.size()) {
            fail("nxs weight size does not match the model");
        }
        if (!weights[index].values.empty()) {
            std::memcpy(parameter->data.data(), weights[index].values.data(), weights[index].values.size() * sizeof(float));
        }
    }
}

std::shared_ptr<nn::Module> module_from_spec(const LayerSpec& spec) { return make_one(spec); }

}  // namespace nxs
}  // namespace nexus
