#pragma once

/**
 * @file nxs_format.hpp
 * @brief Tasinabilir model dosyasi NXS2. NXS1 yalniz egitmen durumudur.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus_model/core/parameter.hpp"
#include "nexus_optim/core/optimizer_state.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace nexus {
namespace nxs {

enum class LayerKind : std::uint16_t {
    Linear = 1,
    Conv2d = 2,
    MultiHeadAttention = 3,
    Lstm = 4,
    Gru = 5,
    TransformerEncoder = 6,
    Embedding = 7,
    BatchNorm1d = 8,
    LayerNorm = 9,
    Relu = 10,
    Swish = 11,
    Dropout = 12,
    MaxPool2d = 13,
    AvgPool2d = 14,
    GlobalAvgPool2d = 15,
    Flatten = 16,
    BatchNorm2d = 17,
    Conv1d = 18,
    Conv3d = 19,
    DepthwiseConv2d = 20,
    GroupNorm = 21,
    InstanceNorm = 22,
    Residual = 23,
    Reshape = 24,
    Rnn = 25,
    PositionalEncoding = 26,
    TransformerDecoder = 27,
    Sequential = 28,
    NamedActivation = 29,
    PythonModule = 30,
    Graph = 31,
    Concat = 32,
    CausalBlock = 33,
    DecoderBlock = 34,   // FAZ 6: nexus::lm (on-norm, RoPE, GQA, SwiGLU)
    LanguageModel = 35,  // FAZ 6: kok; config JSON gomulu, nexus::lm::LanguageModel::load_nxs
    MoEFFN = 36,         // FAZ 14: NexusMoE (yonlendirici + E SwiGLU uzmani); DecoderBlock cocugu
};

enum class AttrKind : std::uint8_t { I32 = 1, F32 = 2, Bool = 3, String = 4 };

struct Attr {
    std::string key;
    AttrKind kind = AttrKind::I32;
    std::int32_t i = 0;
    float f = 0.0f;
    bool b = false;
    std::string s;
};

struct LayerSpec {
    std::uint16_t type = 0;
    std::string name;
    std::vector<Attr> attrs;
    std::vector<LayerSpec> children;
};

struct Weight {
    std::string name;
    std::vector<std::uint64_t> dims;
    std::vector<float> values;
};

struct TrainerRecord {
    std::int32_t epoch = 0;
    std::int32_t global_step = 0;
    double current_loss = 0.0;
    double best_metric = 0.0;
    bool should_stop = false;
    std::vector<std::pair<std::string, double>> metrics;
};

struct MlpBlob {
    std::string name;
    std::vector<std::int32_t> sizes;
    std::uint64_t adam_step = 0;
    std::vector<float> parameters;
    std::vector<float> moments;
};

struct RlRecord {
    std::string kind;
    std::vector<Attr> config;
    std::vector<MlpBlob> nets;
    std::vector<float> elos;
    std::vector<std::vector<float>> player_weights;
};

struct Archive {
    std::vector<LayerSpec> layers;
    std::vector<Weight> weights;
    bool has_optim = false;
    nexus_optim::OptimizerState optim;
    bool has_trainer = false;
    TrainerRecord trainer;
    bool has_rl = false;
    RlRecord rl;
    std::vector<std::uint8_t> python_bytes;
};

inline Attr i32(std::string key, int value) {
    Attr attr;
    attr.key = std::move(key);
    attr.kind = AttrKind::I32;
    attr.i = static_cast<std::int32_t>(value);
    return attr;
}

inline Attr f32(std::string key, float value) {
    Attr attr;
    attr.key = std::move(key);
    attr.kind = AttrKind::F32;
    attr.f = value;
    return attr;
}

inline Attr boolean(std::string key, bool value) {
    Attr attr;
    attr.key = std::move(key);
    attr.kind = AttrKind::Bool;
    attr.b = value;
    return attr;
}

inline Attr str(std::string key, std::string value) {
    Attr attr;
    attr.key = std::move(key);
    attr.kind = AttrKind::String;
    attr.s = std::move(value);
    return attr;
}

[[nodiscard]] inline const Attr* find_attr(const LayerSpec& spec, const std::string& key) {
    for (const Attr& attr : spec.attrs) {
        if (attr.key == key) {
            return &attr;
        }
    }
    return nullptr;
}

[[nodiscard]] inline int require_i32(const LayerSpec& spec, const std::string& key) {
    const Attr* attr = find_attr(spec, key);
    if (attr == nullptr || attr->kind != AttrKind::I32) {
        throw std::runtime_error("nxs attribute missing: " + key);
    }
    return attr->i;
}

[[nodiscard]] inline float require_f32(const LayerSpec& spec, const std::string& key) {
    const Attr* attr = find_attr(spec, key);
    if (attr == nullptr || attr->kind != AttrKind::F32) {
        throw std::runtime_error("nxs attribute missing: " + key);
    }
    return attr->f;
}

[[nodiscard]] inline int attrs_i32(const std::vector<Attr>& attrs, const std::string& key, int fallback) {
    for (const Attr& attr : attrs) {
        if (attr.key == key && attr.kind == AttrKind::I32) {
            return attr.i;
        }
    }
    return fallback;
}

[[nodiscard]] inline float attrs_f32(const std::vector<Attr>& attrs, const std::string& key, float fallback) {
    for (const Attr& attr : attrs) {
        if (attr.key == key && attr.kind == AttrKind::F32) {
            return attr.f;
        }
    }
    return fallback;
}

[[nodiscard]] inline bool require_bool(const LayerSpec& spec, const std::string& key) {
    const Attr* attr = find_attr(spec, key);
    if (attr == nullptr || attr->kind != AttrKind::Bool) {
        throw std::runtime_error("nxs attribute missing: " + key);
    }
    return attr->b;
}

inline LayerSpec spec_of(LayerKind kind, std::string name, std::vector<Attr> attrs = {}) {
    LayerSpec spec;
    spec.type = static_cast<std::uint16_t>(kind);
    spec.name = std::move(name);
    spec.attrs = std::move(attrs);
    return spec;
}

void save_archive(const std::string& path, const Archive& archive);
[[nodiscard]] Archive load_archive(const std::string& path);

[[nodiscard]] std::vector<Weight> collect_weights(const std::vector<nexus_model::Parameter*>& parameters);
void apply_weights(const std::vector<nexus_model::Parameter*>& parameters, const std::vector<Weight>& weights);

}  // namespace nxs

namespace nn {
class Module;
}

namespace nxs {
[[nodiscard]] std::shared_ptr<nn::Module> module_from_spec(const LayerSpec& spec);
void set_python_payload(std::vector<std::uint8_t> bytes);
void register_named_activation(std::string name, std::function<std::shared_ptr<nn::Module>()> factory);
void register_named_module(std::string name, std::function<std::shared_ptr<nn::Module>()> factory);
}

}  // namespace nexus
