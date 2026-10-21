#pragma once

/**
 * @file tape.hpp
 * @brief Mevcut islemlerin gradyanini kaydeder. Yeni matematik yok.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "matrix_pro/cpu/gemm.hpp"
#include "nexus/nn/modules.hpp"

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace nexus {

class Tape {
 public:
    int input(nexus_model::Tensor value) {
        Node node;
        node.op = Op::Input;
        node.value = std::move(value);
        nodes_.push_back(std::move(node));
        return static_cast<int>(nodes_.size()) - 1;
    }

    int add(int left, int right) { return binary(Op::Add, left, right); }
    int mul(int left, int right) { return binary(Op::Mul, left, right); }

    int matmul(int left, int right) {
        const nexus_model::Tensor& a = nodes_.at(static_cast<std::size_t>(left)).value;
        const nexus_model::Tensor& b = nodes_.at(static_cast<std::size_t>(right)).value;
        if (a.rank() != 2 || b.rank() != 2 || a.size(1) != b.size(0)) {
            throw std::invalid_argument("Tape matmul expects 2D shapes that line up");
        }
        const int m = static_cast<int>(a.size(0));
        const int k = static_cast<int>(a.size(1));
        const int n = static_cast<int>(b.size(1));
        nexus_model::Tensor out = nexus_model::Tensor::zeros({a.size(0), b.size(1)});
        matrix_pro::gemm_f32(m, n, k, a.data(), k, b.data(), n, out.data(), n, 1.0f, 0.0f);
        Node node;
        node.op = Op::Matmul;
        node.left = left;
        node.right = right;
        node.value = std::move(out);
        nodes_.push_back(std::move(node));
        return static_cast<int>(nodes_.size()) - 1;
    }

    int relu(int source) {
        const nexus_model::Tensor& in = nodes_.at(static_cast<std::size_t>(source)).value;
        nexus_model::Tensor out = nexus_model::Tensor::zeros(in.shape_vec());
        for (std::size_t i = 0; i < in.numel(); ++i) {
            out[i] = in[i] > 0.0f ? in[i] : 0.0f;
        }
        Node node;
        node.op = Op::Relu;
        node.left = source;
        node.value = std::move(out);
        nodes_.push_back(std::move(node));
        return static_cast<int>(nodes_.size()) - 1;
    }

    int apply(std::shared_ptr<nn::Module> module, int source) {
        if (!module) {
            throw std::invalid_argument("Tape apply requires a module");
        }
        Node node;
        node.op = Op::Module;
        node.left = source;
        node.module = std::move(module);
        node.value = node.module->forward(nodes_.at(static_cast<std::size_t>(source)).value);
        nodes_.push_back(std::move(node));
        return static_cast<int>(nodes_.size()) - 1;
    }

    void backward(int root) {
        Node& node = nodes_.at(static_cast<std::size_t>(root));
        if (node.grad.numel() != node.value.numel()) {
            node.grad = nexus_model::Tensor::zeros(node.value.shape_vec());
            node.grad.fill(1.0f);
        }
        for (int index = root; index >= 0; --index) {
            propagate(index);
        }
    }

    [[nodiscard]] nexus_model::Tensor value(int index) const { return nodes_.at(static_cast<std::size_t>(index)).value; }
    [[nodiscard]] nexus_model::Tensor grad(int index) const { return nodes_.at(static_cast<std::size_t>(index)).grad; }

 private:
    enum class Op { Input, Add, Mul, Matmul, Relu, Module };

    struct Node {
        Op op = Op::Input;
        nexus_model::Tensor value;
        nexus_model::Tensor grad;
        int left = -1;
        int right = -1;
        std::shared_ptr<nn::Module> module;
    };

    int binary(Op op, int left, int right) {
        const nexus_model::Tensor& a = nodes_.at(static_cast<std::size_t>(left)).value;
        const nexus_model::Tensor& b = nodes_.at(static_cast<std::size_t>(right)).value;
        if (a.numel() != b.numel()) {
            throw std::invalid_argument("Tape operands differ in length");
        }
        nexus_model::Tensor out = nexus_model::Tensor::zeros(a.shape_vec());
        for (std::size_t i = 0; i < a.numel(); ++i) {
            out[i] = op == Op::Add ? a[i] + b[i] : a[i] * b[i];
        }
        Node node;
        node.op = op;
        node.left = left;
        node.right = right;
        node.value = std::move(out);
        nodes_.push_back(std::move(node));
        return static_cast<int>(nodes_.size()) - 1;
    }

    void accumulate(int index, const nexus_model::Tensor& extra) {
        if (index < 0) {
            return;
        }
        Node& node = nodes_[static_cast<std::size_t>(index)];
        if (node.grad.numel() != extra.numel()) {
            node.grad = nexus_model::Tensor::zeros(extra.shape_vec());
        }
        for (std::size_t i = 0; i < extra.numel(); ++i) {
            node.grad[i] += extra[i];
        }
    }

    void propagate(int index) {
        Node& node = nodes_[static_cast<std::size_t>(index)];
        if (node.grad.numel() == 0 || node.op == Op::Input) {
            return;
        }
        if (node.op == Op::Add) {
            accumulate(node.left, node.grad);
            accumulate(node.right, node.grad);
            return;
        }
        if (node.op == Op::Mul) {
            const nexus_model::Tensor& a = nodes_[static_cast<std::size_t>(node.left)].value;
            const nexus_model::Tensor& b = nodes_[static_cast<std::size_t>(node.right)].value;
            nexus_model::Tensor ga = nexus_model::Tensor::zeros(a.shape_vec());
            nexus_model::Tensor gb = nexus_model::Tensor::zeros(b.shape_vec());
            for (std::size_t i = 0; i < a.numel(); ++i) {
                ga[i] = node.grad[i] * b[i];
                gb[i] = node.grad[i] * a[i];
            }
            accumulate(node.left, ga);
            accumulate(node.right, gb);
            return;
        }
        if (node.op == Op::Relu) {
            const nexus_model::Tensor& in = nodes_[static_cast<std::size_t>(node.left)].value;
            nexus_model::Tensor gin = nexus_model::Tensor::zeros(in.shape_vec());
            for (std::size_t i = 0; i < in.numel(); ++i) {
                gin[i] = in[i] > 0.0f ? node.grad[i] : 0.0f;
            }
            accumulate(node.left, gin);
            return;
        }
        if (node.op == Op::Module) {
            accumulate(node.left, node.module->backward(node.grad));
            return;
        }
        const nexus_model::Tensor& a = nodes_[static_cast<std::size_t>(node.left)].value;
        const nexus_model::Tensor& b = nodes_[static_cast<std::size_t>(node.right)].value;
        const int m = static_cast<int>(a.size(0));
        const int k = static_cast<int>(a.size(1));
        const int n = static_cast<int>(b.size(1));
        std::vector<float> bt(static_cast<std::size_t>(n * k));
        std::vector<float> at(static_cast<std::size_t>(k * m));
        for (int row = 0; row < k; ++row) {
            for (int col = 0; col < n; ++col) {
                bt[static_cast<std::size_t>(col * k + row)] = b[static_cast<std::size_t>(row * n + col)];
            }
        }
        for (int row = 0; row < m; ++row) {
            for (int col = 0; col < k; ++col) {
                at[static_cast<std::size_t>(col * m + row)] = a[static_cast<std::size_t>(row * k + col)];
            }
        }
        nexus_model::Tensor ga = nexus_model::Tensor::zeros(a.shape_vec());
        nexus_model::Tensor gb = nexus_model::Tensor::zeros(b.shape_vec());
        matrix_pro::gemm_f32(m, k, n, node.grad.data(), n, bt.data(), k, ga.data(), k, 1.0f, 0.0f);
        matrix_pro::gemm_f32(k, n, m, at.data(), m, node.grad.data(), n, gb.data(), n, 1.0f, 0.0f);
        accumulate(node.left, ga);
        accumulate(node.right, gb);
    }

    std::vector<Node> nodes_;
};

}  // namespace nexus
