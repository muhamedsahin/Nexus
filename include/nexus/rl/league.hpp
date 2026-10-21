#pragma once

/**
 * @file league.hpp
 * @brief ELO ve tarihsel politika anlik goruntusu.
 * Skor 1 galibiyet, 0.5 beraberlik, 0 maglubiyettir.
 */

#include "nexus/model/nxs_format.hpp"
#include "nexus/rl/envs.hpp"

#include <string>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace nexus {
namespace rl {

class League {
 public:
    int add_player(float elo = 1000.0f) {
        players_.push_back(Player{elo, {}, false});
        return static_cast<int>(players_.size() - 1);
    }

    void save(int player, std::vector<float> weights) {
        Player& slot = at(player);
        slot.weights = std::move(weights);
        slot.has_weights = true;
    }

    [[nodiscard]] const std::vector<float>& weights(int player) const {
        const Player& slot = at(player);
        if (!slot.has_weights) {
            throw std::logic_error("League player has no checkpoint");
        }
        return slot.weights;
    }

    void result(int first, int second, float score_first, float k = 32.0f) {
        Player& a = at(first);
        Player& b = at(second);
        const float expected = 1.0f / (1.0f + std::pow(10.0f, (b.elo - a.elo) / 400.0f));
        a.elo += k * (score_first - expected);
        b.elo += k * ((1.0f - score_first) - (1.0f - expected));
    }

    [[nodiscard]] float elo(int player) const { return at(player).elo; }
    [[nodiscard]] int size() const { return static_cast<int>(players_.size()); }

    void save_nxs(const std::string& path) const {
        nxs::Archive archive;
        archive.has_rl = true;
        archive.rl.kind = "league";
        for (const Player& player : players_) {
            archive.rl.elos.push_back(player.elo);
            archive.rl.player_weights.push_back(player.has_weights ? player.weights : std::vector<float>{});
        }
        nxs::save_archive(path, archive);
    }

    [[nodiscard]] static League load_nxs(const std::string& path) {
        const nxs::Archive archive = nxs::load_archive(path);
        if (!archive.has_rl || archive.rl.kind != "league") {
            throw std::runtime_error("nxs file is not a league");
        }
        League league;
        for (std::size_t i = 0; i < archive.rl.elos.size(); ++i) {
            const int id = league.add_player(archive.rl.elos[i]);
            if (i < archive.rl.player_weights.size() && !archive.rl.player_weights[i].empty()) {
                league.save(id, archive.rl.player_weights[i]);
            }
        }
        return league;
    }

 private:
    struct Player {
        float elo = 1000.0f;
        std::vector<float> weights;
        bool has_weights = false;
    };

    Player& at(int player) {
        if (player < 0 || player >= static_cast<int>(players_.size())) {
            throw std::invalid_argument("League player id is outside the table");
        }
        return players_[static_cast<std::size_t>(player)];
    }

    const Player& at(int player) const {
        if (player < 0 || player >= static_cast<int>(players_.size())) {
            throw std::invalid_argument("League player id is outside the table");
        }
        return players_[static_cast<std::size_t>(player)];
    }

    std::vector<Player> players_;
};

inline float scripted_duel(Duel& env, int action_a, int action_b) {
    env.reset(1);
    std::unordered_map<AgentID, Tensor> actions;
    actions.emplace("a", discrete_action_tensor(action_a));
    actions.emplace("b", discrete_action_tensor(action_b));
    return env.step(actions).rewards.at("a");
}

}  // namespace rl
}  // namespace nexus
