#pragma once

/**
 * @file module.hpp
 * @brief Egitilebilir parametre tasiyan kullanici katmani.
 *
 * `add_parameter` hem bu listede hem Module agacinda kayitlidir, boylece
 * `parameters()`, `zero_grad` ve `state_dict` kaydi gorur.
 * Ileri ve geri govde kullanicinin turettigi siniftadir.
 */

#include "nexus_model/core/module.hpp"

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace nexus {

class CustomModule : public nexus_model::Module {
 public:
    void add_parameter(std::string name, std::shared_ptr<nexus_model::Parameter> parameter) {
        if (name.empty() || !parameter) {
            throw nexus_model::ModelError("CustomModule parameter name and storage are required");
        }
        register_parameter(name, parameter);
        slots_.push_back(Slot{std::move(name), std::move(parameter)});
    }

    void add_child(std::string name, std::shared_ptr<nexus_model::Module> module) {
        if (name.empty() || !module) {
            throw nexus_model::ModelError("CustomModule child name and module are required");
        }
        register_module(std::move(name), std::move(module));
    }

    [[nodiscard]] nexus_model::Parameter* find_parameter(const std::string& name) {
        for (Slot& slot : slots_) {
            if (slot.name == name) {
                return slot.parameter.get();
            }
        }
        return nullptr;
    }

    [[nodiscard]] const nexus_model::Parameter* find_parameter(const std::string& name) const {
        for (const Slot& slot : slots_) {
            if (slot.name == name) {
                return slot.parameter.get();
            }
        }
        return nullptr;
    }

 private:
    struct Slot {
        std::string name;
        std::shared_ptr<nexus_model::Parameter> parameter;
    };

    std::vector<Slot> slots_;
};

}  // namespace nexus
