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
    test_league_nxs();
    if (g_failed != 0) {
        std::cout << g_failed << " kontrol basarisiz\n";
        return 1;
    }
    std::cout << "tum kontroller gecti\n";
    return 0;
}
