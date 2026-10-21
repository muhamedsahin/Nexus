/**
 * @file main.cpp
 * Isbirligi (IPPO ve MAPPO) ile rekabetci oz-oyun (ELO) ayni programda.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/nexus.hpp"

#include <cmath>
#include <iostream>

int main() {
    nexus::rl::MultiAgentTrainer trainer;

    nexus::rl::ButtonWorld buttons;
    const nexus::rl::MarlReport ippo = trainer.train_independent(buttons, 12, 36, 11);
    const nexus::rl::MarlReport mappo = trainer.train_centralized(buttons, 12, 36, 12);
    std::cout << "ippo " << ippo.first_return << " -> " << ippo.last_return << "\n";
    std::cout << "mappo " << mappo.first_return << " -> " << mappo.last_return << "\n";

    nexus::rl::TeamButton trio(3);
    const nexus::rl::MarlReport trio_ippo = trainer.train_independent(trio, 3, 18, 13);
    const nexus::rl::MarlReport trio_mappo = trainer.train_centralized(trio, 3, 18, 14);
    std::cout << "trio_ippo " << trio_ippo.last_return << " trio_mappo " << trio_mappo.last_return << "\n";

    nexus::rl::League league;
    const int learner = league.add_player();
    const int history = league.add_player();
    league.save(learner, {1.0f});
    league.save(history, {0.0f});
    nexus::rl::Duel duel;
    for (int game = 0; game < 8; ++game) {
        const float score = nexus::rl::scripted_duel(duel, 1, 0);
        league.result(learner, history, score > 0.0f ? 1.0f : 0.0f);
    }
    std::cout << "elo learner=" << league.elo(learner) << " history=" << league.elo(history) << "\n";

    const bool learned = ippo.last_return > ippo.first_return + 0.4f && mappo.last_return > mappo.first_return + 0.4f;
    const bool trio_ok = std::isfinite(trio_ippo.last_return) && std::isfinite(trio_mappo.last_return);
    const bool rated = league.elo(learner) != league.elo(history);
    return learned && trio_ok && rated ? 0 : 1;
}
