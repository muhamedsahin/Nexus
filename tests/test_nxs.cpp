/**
 * @file test_nxs.cpp
 * @brief NXS2 arsiv, grafik gidis-donus ve NXS1 reddi.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/model/nxs_format.hpp"
#include "nexus/nn/activations.hpp"
#include "nexus/nn/modules.hpp"
#include "nexus/optim/optimizers.hpp"
#include "nexus/rl/league.hpp"
#include "nexus/rl/mlp.hpp"
#include "nexus/rl/ppo.hpp"
#include "nexus/train/language.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

namespace {

int g_failed = 0;

void expect(bool condition, const char* name) {
    if (condition) {
        std::cout << "  PASS  " << name << "\n";
        return;
    }
    std::cout << "  FAIL  " << name << "\n";
    ++g_failed;
}

std::filesystem::path scratch(const char* name) {
    auto path = std::filesystem::temp_directory_path() / name;
    std::filesystem::remove(path);
    return path;
}

void test_roundtrip_and_reject() {
    const auto path = scratch("nexus_nxs2_empty.nxs");
    nexus::nxs::Archive archive;
    nexus::nxs::LayerSpec layer = nexus::nxs::spec_of(nexus::nxs::LayerKind::Relu, "ReLU");
    archive.layers.push_back(layer);
    nexus::nxs::save_archive(path.string(), archive);
    const nexus::nxs::Archive loaded = nexus::nxs::load_archive(path.string());
    expect(loaded.layers.size() == 1 && loaded.layers[0].type == static_cast<std::uint16_t>(nexus::nxs::LayerKind::Relu),
           "empty graph roundtrip");

    const auto trainer_path = scratch("nexus_trainer_only.nxs");
    {
        std::ofstream out(trainer_path, std::ios::binary);
        out.write("NXS1", 4);
    }
    bool rejected = false;
    try {
        (void)nexus::nxs::load_archive(trainer_path.string());
    } catch (const std::runtime_error& error) {
        rejected = std::string(error.what()).find("egitmen durumu") != std::string::npos;
    }
    expect(rejected, "NXS1 is not a model archive");

    const auto broken = scratch("nexus_truncated.nxs");
    {
        std::ofstream out(broken, std::ios::binary);
        out.write("NXS2", 4);
        const std::uint16_t version = 1;
        out.write(reinterpret_cast<const char*>(&version), sizeof(version));
    }
    bool truncated = false;
    try {
        (void)nexus::nxs::load_archive(broken.string());
    } catch (const std::runtime_error&) {
        truncated = true;
    }
    expect(truncated, "truncated nxs file fails");
}

void test_sequential_logits() {
    auto model = nexus::nn::Sequential().add<nexus::nn::Linear>(2, 4).add(nexus::nn::swish()).add<nexus::nn::Linear>(4, 2);
    nexus_model::Tensor input = nexus_model::Tensor::zeros({1, 2});
    input[0] = 0.5f;
    input[1] = -0.25f;
    nexus_model::Tensor logits = model.forward(input);
    nexus_model::Tensor grad = nexus_model::Tensor::zeros({1, 2});
    grad[0] = 0.1f;
    grad[1] = -0.2f;
    model.backward(grad);
    nexus::optim::AdamW adam(model->parameters(), {.lr = 0.05f});
    adam.step();
    logits = model.forward(input);

    const auto path = scratch("nexus_sequential.nxs");
    model.save_nxs(path.string());
    nexus::nn::Sequential loaded = nexus::nn::Sequential::load_nxs(path.string());
    nexus_model::Tensor again = loaded.forward(input);
    float worst = 0.0f;
    for (std::size_t i = 0; i < logits.numel(); ++i) {
        worst = std::max(worst, std::fabs(logits[i] - again[i]));
    }
    expect(worst < 1.0e-5f, "sequential logits survive nxs");

    auto custom = nexus::nn::Sequential().add(nexus::nn::make_activation("Mine", [](float x) { return x; }));
    const auto lambda_path = scratch("nexus_lambda.nxs");
    custom.save_nxs(lambda_path.string());
    bool rejected = false;
    try {
        (void)nexus::nn::Sequential::load_nxs(lambda_path.string());
    } catch (const std::runtime_error& error) {
        rejected = std::string(error.what()).find("govdesi") != std::string::npos;
    }
    expect(rejected, "python activation without a registered body is rejected");
}

void test_mlp_blob() {
    nexus::rl::Mlp net({2, 4, 2}, 3);
    std::vector<float> input{0.2f, -0.4f};
    std::vector<float> first(2);
    std::vector<float> second(2);
    net.forward(input.data(), 1, first.data());
    const auto blob = net.export_blob("actor");
    nexus::rl::Mlp restored({2, 4, 2}, 99);
    restored.import_blob(blob);
    restored.forward(input.data(), 1, second.data());
    float worst = 0.0f;
    for (int i = 0; i < 2; ++i) {
        worst = std::max(worst, std::fabs(first[static_cast<std::size_t>(i)] - second[static_cast<std::size_t>(i)]));
    }
    expect(worst < 1.0e-5f, "mlp blob restores the forward");
}

void test_ppo_nxs() {
    nexus::rl::LineWorld env;
    nexus::rl::PPOAgent::Config config;
    config.hidden = 8;
    config.seed = 4;
    config.epochs = 1;
    nexus::rl::PPOAgent agent(env.observation_dim(), env.action_count(), config);
    agent.learn(env, 1, 4);
    std::vector<float> observation(static_cast<std::size_t>(env.observation_dim()), 0.1f);
    float log_prob = 0.0f;
    float value = 0.0f;
    const int action = agent.act(observation.data(), log_prob, value, true);
    const auto path = scratch("nexus_ppo.nxs");
    agent.save_nxs(path.string());
    nexus::rl::PPOAgent loaded = nexus::rl::PPOAgent::load_nxs(path.string());
    float log_prob_b = 0.0f;
    float value_b = 0.0f;
    const int again = loaded.act(observation.data(), log_prob_b, value_b, true);
    expect(action == again, "ppo greedy action survives nxs");
}

void test_python_module_without_body() {
    nexus::nxs::Archive archive;
    archive.layers.push_back(nexus::nxs::spec_of(nexus::nxs::LayerKind::PythonModule, "Yok"));
    const auto path = scratch("nexus_python_module.nxs");
    nexus::nxs::save_archive(path.string(), archive);
    bool rejected = false;
    try {
        (void)nexus::nn::Sequential::load_nxs(path.string());
    } catch (const std::runtime_error& error) {
        rejected = std::string(error.what()).find("govdesi") != std::string::npos;
    }
    expect(rejected, "python module without a body is rejected");
}

void identity_linear(nexus::nn::Linear& layer, int width) {
    const std::vector<nexus_model::Parameter*> params = layer.parameters();
    params[0]->data.zero();
    for (int index = 0; index < width; ++index) {
        params[0]->data[static_cast<std::size_t>(index * width + index)] = 1.0f;
    }
    if (params.size() > 1) {
        params[1]->data.zero();
    }
}

void test_graph_merge() {
    auto left = std::make_shared<nexus::nn::Linear>(2, 2, true);
    auto right = std::make_shared<nexus::nn::Linear>(2, 2, true);
    auto joined = std::make_shared<nexus::nn::Linear>(4, 4, true);
    identity_linear(*left, 2);
    identity_linear(*right, 2);
    identity_linear(*joined, 4);
    nexus::nn::Graph graph;
    graph.add("katman2", left);
    graph.add("katman3", std::make_shared<nexus::nn::ReLU>());
    graph.add("katman4", right);
    graph.concat("katman5", joined, {"katman4", "katman2"});
    nexus_model::Tensor input = nexus_model::Tensor::zeros({1, 2});
    input[0] = 3.0f;
    input[1] = 5.0f;
    const nexus_model::Tensor logits = graph.forward(input);
    const float expected[4] = {3.0f, 5.0f, 3.0f, 5.0f};
    float worst = 0.0f;
    expect(logits.numel() == 4, "graph concat width");
    for (int index = 0; index < 4; ++index) {
        worst = std::max(worst, std::fabs(logits[static_cast<std::size_t>(index)] - expected[index]));
    }
    expect(worst < 1.0e-5f, "layer 5 sees layer 4 and layer 2");
    const auto path = scratch("nexus_graph.nxs");
    graph.save_nxs(path.string());
    nexus::nn::Graph loaded = nexus::nn::Graph::load_nxs(path.string());
    const nexus_model::Tensor again = loaded.forward(input);
    worst = 0.0f;
    for (int index = 0; index < 4; ++index) {
        worst = std::max(worst, std::fabs(again[static_cast<std::size_t>(index)] - expected[index]));
    }
    expect(worst < 1.0e-5f, "graph nxs restores both inputs");
    bool unknown = false;
    try {
        nexus::nn::Graph broken;
        broken.add("katman2", std::make_shared<nexus::nn::ReLU>());
        broken.add("katman5", std::make_shared<nexus::nn::ReLU>(), {"katman2", "yok"}, false);
    } catch (const std::invalid_argument&) {
        unknown = true;
    }
    expect(unknown, "unknown graph input is rejected");

    auto sum_layer = std::make_shared<nexus::nn::Linear>(2, 2, true);
    identity_linear(*sum_layer, 2);
    auto add_left = std::make_shared<nexus::nn::Linear>(2, 2, true);
    auto add_right = std::make_shared<nexus::nn::Linear>(2, 2, true);
    identity_linear(*add_left, 2);
    identity_linear(*add_right, 2);
    nexus::nn::Graph added;
    added.add("katman2", add_left);
    added.add("katman4", add_right);
    added.add("katman5", sum_layer, {"katman4", "katman2"}, false);
    const nexus_model::Tensor summed = added.forward(input);
    worst = 0.0f;
    for (int index = 0; index < 2; ++index) {
        const float want = (index == 0 ? 6.0f : 10.0f);
        worst = std::max(worst, std::fabs(summed[static_cast<std::size_t>(index)] - want));
    }
    expect(worst < 1.0e-5f, "graph add sums both inputs");
}

void test_causal_mask() {
    nexus::nn::CausalTransformerBlock block(4, 1, 8);
    block.eval();
    nexus_model::Tensor input = nexus_model::Tensor::zeros({1, 3, 4});
    for (std::size_t index = 0; index < input.numel(); ++index) {
        input[index] = 0.1f;
    }
    const nexus_model::Tensor first = block.forward(input).clone();
    input[8] = 9.0f;
    const nexus_model::Tensor second = block.forward(input);
    float past = 0.0f;
    for (int index = 0; index < 8; ++index) {
        past = std::max(past, std::fabs(first[static_cast<std::size_t>(index)] - second[static_cast<std::size_t>(index)]));
    }
    float future = 0.0f;
    for (int index = 8; index < 12; ++index) {
        future = std::max(future, std::fabs(first[static_cast<std::size_t>(index)] - second[static_cast<std::size_t>(index)]));
    }
    expect(past < 1.0e-5f, "future token does not change past positions");
    expect(future > 1.0e-4f, "future token changes its own position");

    nexus::nn::Sequential net;
    net.add(std::make_shared<nexus::nn::CausalTransformerBlock>(4, 1, 8));
    const nexus_model::Tensor before = net.forward(input).clone();
    const auto path = scratch("nexus_causal.nxs");
    net.save_nxs(path.string());
    nexus::nn::Sequential loaded = nexus::nn::Sequential::load_nxs(path.string());
    const nexus_model::Tensor after = loaded.forward(input);
    float worst = 0.0f;
    for (std::size_t index = 0; index < before.numel(); ++index) {
        worst = std::max(worst, std::fabs(before[index] - after[index]));
    }
    expect(worst < 1.0e-5f, "causal block nxs roundtrip");
}

void test_language_loss_drops() {
    const int vocab = 6;
    const int embed = 8;
    nexus::nn::Sequential model;
    model.add(std::make_shared<nexus::nn::Embedding>(vocab, embed, -1));
    model.add(std::make_shared<nexus::nn::PositionalEncoding>(embed, 32, false));
    model.add(std::make_shared<nexus::nn::CausalTransformerBlock>(embed, 2, 16));
    model.add(std::make_shared<nexus::nn::Linear>(static_cast<std::size_t>(embed), static_cast<std::size_t>(vocab), true));
    nexus::LanguageTrainer trainer(model, 0.08f);
    nexus::TokenRows rows;
    rows.batch = 4;
    rows.time = 8;
    rows.values.assign(static_cast<std::size_t>(rows.batch * rows.time), 0);
    for (int batch = 0; batch < rows.batch; ++batch) {
        for (int time = 0; time < rows.time; ++time) {
            rows.values[static_cast<std::size_t>(batch * rows.time + time)] = time % vocab;
        }
    }
    const float first = trainer.fit(rows, 1);
    const float later = trainer.fit(rows, 30);
    expect(later < first, "shifted token loss decreases");
}

void test_league_nxs() {
    nexus::rl::League league;
    const int player = league.add_player(1100.0f);
    league.save(player, {0.25f, -0.5f, 1.0f});
    const auto path = scratch("nexus_league.nxs");
    league.save_nxs(path.string());
    nexus::rl::League loaded = nexus::rl::League::load_nxs(path.string());
    expect(loaded.size() == 1 && std::fabs(loaded.elo(0) - 1100.0f) < 1.0e-3f && loaded.weights(0).size() == 3, "league nxs");
}

}  // namespace

int main() {
    std::cout << "Nexus NXS testi\n";
    test_roundtrip_and_reject();
    test_sequential_logits();
    test_mlp_blob();
    test_ppo_nxs();
    test_python_module_without_body();
    test_graph_merge();
    test_causal_mask();
    test_language_loss_drops();
    test_league_nxs();
    if (g_failed != 0) {
        std::cout << g_failed << " kontrol basarisiz\n";
        return 1;
    }
    std::cout << "tum kontroller gecti\n";
    return 0;
}
