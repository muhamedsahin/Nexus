#pragma once

/**
 * @file microtape.hpp
 * @brief Skaler ters-mod teyp. Ozel kayip yalnizca ileri ifadeyi kurar;
 * `backward` kokun turevini girislere yayar.
 *
 * Desteklenen op'lar: toplama, cikarma, carpma, bolme, negatif, kare,
 * exp, log, mutlak deger, max. Log pozitif arguman ister. Bolum sifira bolmez.
 * Esit max'ta sol dal tum gradyani alir.
 */

#include "nexus/core/error.hpp"

#include <cmath>
#include <vector>

namespace nexus {

class MicroTape {
 public:
    struct Var {
        int index = -1;
    };

    Var constant(float value) { return push(Op::Const, -1, -1, value, false); }

    /// Gradyani okunabilen girdi. Deger ileri geciste sabittir.
    Var input(float value) { return push(Op::Const, -1, -1, value, true); }

    Var add(Var lhs, Var rhs) { return binary(Op::Add, lhs, rhs, value(lhs) + value(rhs)); }
    Var sub(Var lhs, Var rhs) { return binary(Op::Sub, lhs, rhs, value(lhs) - value(rhs)); }
    Var mul(Var lhs, Var rhs) { return binary(Op::Mul, lhs, rhs, value(lhs) * value(rhs)); }
    Var neg(Var lhs) { return push(Op::Neg, lhs.index, -1, -value(lhs), false); }
    Var square(Var lhs) {
        const float v = value(lhs);
        return push(Op::Square, lhs.index, -1, v * v, false);
    }
    Var exp(Var lhs) { return push(Op::Exp, lhs.index, -1, std::exp(value(lhs)), false); }

    Var div(Var lhs, Var rhs) {
        const float denominator = value(rhs);
        if (denominator == 0.0f) {
            throw TensorError("MicroTape: sifira bolme");
        }
        return binary(Op::Div, lhs, rhs, value(lhs) / denominator);
    }

    Var log(Var lhs) {
        const float v = value(lhs);
        if (!(v > 0.0f)) {
            throw TensorError("MicroTape: log argumani pozitif olmali");
        }
        return push(Op::Log, lhs.index, -1, std::log(v), false);
    }

    Var abs(Var lhs) { return push(Op::Abs, lhs.index, -1, std::fabs(value(lhs)), false); }

    Var max(Var lhs, Var rhs) {
        const float a = value(lhs);
        const float b = value(rhs);
        return binary(Op::Max, lhs, rhs, a > b ? a : b);
    }

    void backward(Var root) {
        if (root.index < 0 || static_cast<std::size_t>(root.index) >= nodes_.size()) {
            throw TensorError("MicroTape: gecersiz kok");
        }
        for (Node& node : nodes_) {
            node.grad = 0.0f;
        }
        nodes_[static_cast<std::size_t>(root.index)].grad = 1.0f;
        for (int i = root.index; i >= 0; --i) {
            const Node& node = nodes_[static_cast<std::size_t>(i)];
            const float g = node.grad;
            if (g == 0.0f || node.lhs < 0) {
                continue;
            }
            Node& left = nodes_[static_cast<std::size_t>(node.lhs)];
            switch (node.op) {
                case Op::Add:
                    left.grad += g;
                    nodes_[static_cast<std::size_t>(node.rhs)].grad += g;
                    break;
                case Op::Sub:
                    left.grad += g;
                    nodes_[static_cast<std::size_t>(node.rhs)].grad -= g;
                    break;
                case Op::Mul:
                    left.grad += g * nodes_[static_cast<std::size_t>(node.rhs)].value;
                    nodes_[static_cast<std::size_t>(node.rhs)].grad += g * left.value;
                    break;
                case Op::Div: {
                    const float b = nodes_[static_cast<std::size_t>(node.rhs)].value;
                    left.grad += g / b;
                    nodes_[static_cast<std::size_t>(node.rhs)].grad += g * (-left.value / (b * b));
                    break;
                }
                case Op::Neg:
                    left.grad -= g;
                    break;
                case Op::Square:
                    left.grad += g * (2.0f * left.value);
                    break;
                case Op::Exp:
                    left.grad += g * node.value;
                    break;
                case Op::Log:
                    left.grad += g / left.value;
                    break;
                case Op::Abs:
                    if (left.value > 0.0f) {
                        left.grad += g;
                    } else if (left.value < 0.0f) {
                        left.grad -= g;
                    }
                    break;
                case Op::Max:
                    if (left.value > nodes_[static_cast<std::size_t>(node.rhs)].value) {
                        left.grad += g;
                    } else if (left.value < nodes_[static_cast<std::size_t>(node.rhs)].value) {
                        nodes_[static_cast<std::size_t>(node.rhs)].grad += g;
                    } else {
                        left.grad += g;
                    }
                    break;
                case Op::Const:
                    break;
            }
        }
    }

    [[nodiscard]] float value(Var var) const { return nodes_.at(static_cast<std::size_t>(var.index)).value; }
    [[nodiscard]] float grad(Var var) const { return nodes_.at(static_cast<std::size_t>(var.index)).grad; }
    [[nodiscard]] std::size_t size() const noexcept { return nodes_.size(); }

    void reset() { nodes_.clear(); }

 private:
    enum class Op { Const, Add, Sub, Mul, Div, Neg, Square, Exp, Log, Abs, Max };

    struct Node {
        Op op = Op::Const;
        int lhs = -1;
        int rhs = -1;
        float value = 0.0f;
        float grad = 0.0f;
        bool tracked = false;
    };

    Var push(Op op, int lhs, int rhs, float value, bool tracked) {
        Node node;
        node.op = op;
        node.lhs = lhs;
        node.rhs = rhs;
        node.value = value;
        node.tracked = tracked;
        nodes_.push_back(node);
        return Var{static_cast<int>(nodes_.size() - 1)};
    }

    Var binary(Op op, Var lhs, Var rhs, float value) {
        require(lhs);
        require(rhs);
        return push(op, lhs.index, rhs.index, value, false);
    }

    void require(Var var) const {
        if (var.index < 0 || static_cast<std::size_t>(var.index) >= nodes_.size()) {
            throw TensorError("MicroTape: gecersiz dugum");
        }
    }

    std::vector<Node> nodes_;
};

}  // namespace nexus
