/**
 * @file test_phase2_hardware.cpp
 * @brief CPU GEMM, halka all-reduce, HLO ve hibrit CPU+GPU esitligi.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "matrix_pro/detail/cuda_hooks.hpp"
#include "matrix_pro/hardware.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

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

float max_abs(const std::vector<float>& left, const std::vector<float>& right) {
    float worst = 0.0f;
    for (std::size_t i = 0; i < left.size(); ++i) {
        worst = std::max(worst, std::fabs(left[i] - right[i]));
    }
    return worst;
}

void fill_pattern(std::vector<float>& values) {
    for (std::size_t i = 0; i < values.size(); ++i) {
        values[i] = static_cast<float>(static_cast<int>(i * 17 % 100)) * 0.001f - 0.05f;
    }
}

void naive_sum(std::vector<std::vector<float>>& ranks) {
    std::vector<float> acc(ranks[0].size(), 0.0f);
    for (const std::vector<float>& rank : ranks) {
        for (std::size_t i = 0; i < acc.size(); ++i) {
            acc[i] += rank[i];
        }
    }
    for (std::vector<float>& rank : ranks) {
        rank = acc;
    }
}

void test_cpu_gemm() {
    std::cout << "cpu gemm\n";
    const int m = 67;
    const int n = 45;
    const int k = 33;
    std::vector<float> a(static_cast<std::size_t>(m * k));
    std::vector<float> b(static_cast<std::size_t>(k * n));
    std::vector<float> fast(static_cast<std::size_t>(m * n), 0.0f);
    std::vector<float> ref(fast);
    fill_pattern(a);
    fill_pattern(b);
    matrix_pro::gemm_f32(m, n, k, a.data(), k, b.data(), n, fast.data(), n, 1.0f, 0.0f);
    matrix_pro::gemm_reference(m, n, k, a.data(), k, b.data(), n, ref.data(), n, 1.0f, 0.0f);
    const float error = max_abs(fast, ref);
    std::cout << "  info  max_abs=" << error << "\n";
    expect(error < 1.0e-5f, "AVX/scalar GEMM < 1e-5");

    std::vector<float> x(128), y(128), z(128);
    fill_pattern(x);
    fill_pattern(y);
    matrix_pro::elementwise_add(x.data(), y.data(), z.data(), 128);
    bool add_ok = true;
    for (int i = 0; i < 128; ++i) {
        if (std::fabs(z[static_cast<std::size_t>(i)] - (x[static_cast<std::size_t>(i)] + y[static_cast<std::size_t>(i)])) > 1.0e-6f) {
            add_ok = false;
        }
    }
    expect(add_ok, "elementwise add");
    matrix_pro::elementwise_relu(x.data(), z.data(), 128);
    expect(z[0] >= 0.0f, "relu non-negative");
}

void test_collectives() {
    std::cout << "collectives\n";
    constexpr int kRanks = 4;
    constexpr int kCount = 16;
    std::vector<std::vector<float>> initial(kRanks, std::vector<float>(kCount));
    for (int rank = 0; rank < kRanks; ++rank) {
        for (int i = 0; i < kCount; ++i) {
            initial[static_cast<std::size_t>(rank)][static_cast<std::size_t>(i)] = static_cast<float>(rank + 1);
        }
    }
    auto run = [&](matrix_pro::CollectiveAlgorithm algorithm, matrix_pro::ReduceOp op, const char* name) {
        std::vector<std::vector<float>> data = initial;
        std::vector<matrix_pro::CollectiveBuffer> buffers(kRanks);
        for (int rank = 0; rank < kRanks; ++rank) {
            buffers[static_cast<std::size_t>(rank)].data = data[static_cast<std::size_t>(rank)].data();
            buffers[static_cast<std::size_t>(rank)].count = kCount;
            buffers[static_cast<std::size_t>(rank)].device = matrix_pro::DeviceId::cpu();
        }
        matrix_pro::all_reduce(buffers.data(), kRanks, op, algorithm);
        std::vector<std::vector<float>> oracle = initial;
        naive_sum(oracle);
        if (op == matrix_pro::ReduceOp::Mean) {
            for (std::vector<float>& rank : oracle) {
                for (float& value : rank) {
                    value /= static_cast<float>(kRanks);
                }
            }
        } else if (op == matrix_pro::ReduceOp::Max) {
            for (std::vector<float>& rank : oracle) {
                std::fill(rank.begin(), rank.end(), 4.0f);
            }
        }
        float worst = 0.0f;
        for (int rank = 0; rank < kRanks; ++rank) {
            worst = std::max(worst, max_abs(data[static_cast<std::size_t>(rank)], oracle[static_cast<std::size_t>(rank)]));
        }
        expect(worst == 0.0f, name);
    };
    run(matrix_pro::CollectiveAlgorithm::Ring, matrix_pro::ReduceOp::Sum, "ring sum");
    run(matrix_pro::CollectiveAlgorithm::Ring, matrix_pro::ReduceOp::Mean, "ring mean");
    run(matrix_pro::CollectiveAlgorithm::Ring, matrix_pro::ReduceOp::Max, "ring max");
    run(matrix_pro::CollectiveAlgorithm::Tree, matrix_pro::ReduceOp::Sum, "tree sum");

    std::vector<std::vector<float>> odd(3, std::vector<float>(7, 1.0f));
    odd[1][0] = 5.0f;
    std::vector<matrix_pro::CollectiveBuffer> odd_buffers(3);
    for (int rank = 0; rank < 3; ++rank) {
        odd_buffers[static_cast<std::size_t>(rank)].data = odd[static_cast<std::size_t>(rank)].data();
        odd_buffers[static_cast<std::size_t>(rank)].count = 7;
        odd_buffers[static_cast<std::size_t>(rank)].device = matrix_pro::DeviceId::cpu();
    }
    matrix_pro::all_reduce(odd_buffers.data(), 3, matrix_pro::ReduceOp::Sum, matrix_pro::CollectiveAlgorithm::Ring);
    expect(odd[0][0] == 7.0f && odd[2][1] == 3.0f, "ring tail");

    std::vector<float> root{1, 2, 3, 4, 5, 6};
    std::vector<float> other_a(2, 0), other_b(2, 0);
    matrix_pro::CollectiveBuffer scatter_ranks[3] = {
        {root.data(), root.size(), matrix_pro::DeviceId::cpu()},
        {other_a.data(), other_a.size(), matrix_pro::DeviceId::cpu()},
        {other_b.data(), other_b.size(), matrix_pro::DeviceId::cpu()},
    };
    matrix_pro::scatter(scatter_ranks, 3, 0, 2);
    expect(root[0] == 1.0f && root[1] == 2.0f && other_a[0] == 3.0f && other_b[1] == 6.0f, "scatter");
}

void test_hlo_and_pipeline() {
    std::cout << "hlo and pipeline\n";
    const char* text =
        "HloModule gemm\n"
        "ENTRY main {\n"
        "  a = f32[8,4] parameter(0)\n"
        "  b = f32[4,6] parameter(1)\n"
        "  ROOT c = f32[8,6] dot(a, b), lhs_contracting_dims={1}, rhs_contracting_dims={0}\n"
        "}\n";
    matrix_pro::HloExecutable executable = matrix_pro::compile_hlo(text);
    std::vector<float> a(32), b(24), c(48, 0), ref(48, 0);
    fill_pattern(a);
    fill_pattern(b);
    executable.run(a.data(), b.data(), c.data());
    matrix_pro::gemm_reference(8, 6, 4, a.data(), 4, b.data(), 6, ref.data(), 6, 1.0f, 0.0f);
    expect(max_abs(c, ref) < 1.0e-5f, "HLO dot");

    constexpr int kRows = 4;
    constexpr int kStride = 4;
    constexpr int kBatches = 3;
    std::vector<float> data(static_cast<std::size_t>(kBatches * kRows * kStride), 1.0f);
    auto scale = [](int, float* view, int stride, int rows, void*) {
        for (int row = 0; row < rows; ++row) {
            for (int col = 0; col < stride; ++col) {
                view[row * stride + col] *= 2.0f;
            }
        }
    };
    auto add = [](int, float* view, int stride, int rows, void*) {
        for (int row = 0; row < rows; ++row) {
            for (int col = 0; col < stride; ++col) {
                view[row * stride + col] += 1.0f;
            }
        }
    };
    matrix_pro::PipelineStage stages[2] = {
        {matrix_pro::DeviceId::cpu(), scale, nullptr},
        {matrix_pro::DeviceId::cpu(), add, nullptr},
    };
    matrix_pro::HeterogeneousScheduler::instance().run_pipeline(stages, 2, data.data(), kStride, kRows, kBatches);
    expect(data.front() == 3.0f && data.back() == 3.0f, "pipeline scale then add");
}

void test_cuda_and_hybrid() {
#if !defined(MATRIX_PRO_ENABLE_CUDA)
    std::cout << "cuda kapali, gpu adimlari atlandi\n";
    expect(matrix_pro::DeviceManager::instance().count(matrix_pro::DeviceType::Cpu) == 1, "cpu device");
#else
    std::cout << "cuda and hybrid\n";
    const int devices = matrix_pro::hw::cuda_device_count();
    std::cout << "  info  cuda_devices=" << devices << "\n";
    expect(matrix_pro::DeviceManager::instance().count(matrix_pro::DeviceType::Cpu) == 1, "cpu device");
    if (devices <= 0) {
        expect(true, "cuda device yok");
        return;
    }
    const int m = 64;
    const int n = 48;
    const int k = 32;
    std::vector<float> a(static_cast<std::size_t>(m * k));
    std::vector<float> b(static_cast<std::size_t>(k * n));
    std::vector<float> gpu(static_cast<std::size_t>(m * n), 0.0f);
    std::vector<float> ref(gpu);
    fill_pattern(a);
    fill_pattern(b);
    float* device_a = matrix_pro::hw::cuda_alloc(0, a.size());
    float* device_b = matrix_pro::hw::cuda_alloc(0, b.size());
    float* device_c = matrix_pro::hw::cuda_alloc(0, gpu.size());
    matrix_pro::hw::cuda_copy_h2d(0, device_a, a.data(), a.size());
    matrix_pro::hw::cuda_copy_h2d(0, device_b, b.data(), b.size());
    matrix_pro::hw::cuda_gemm(0, m, n, k, device_a, k, device_b, n, device_c, n, 1.0f, 0.0f);
    matrix_pro::hw::cuda_copy_d2h(0, gpu.data(), device_c, gpu.size());
    matrix_pro::hw::cuda_sync(0);
    matrix_pro::gemm_reference(m, n, k, a.data(), k, b.data(), n, ref.data(), n, 1.0f, 0.0f);
    const float gpu_error = max_abs(gpu, ref);
    std::cout << "  info  gpu_max_abs=" << gpu_error << "\n";
    expect(gpu_error < 1.0e-5f, "GPU GEMM < 1e-5");
    matrix_pro::hw::cuda_free(0, device_a);
    matrix_pro::hw::cuda_free(0, device_b);
    matrix_pro::hw::cuda_free(0, device_c);

    constexpr int kSimRanks = 4;
    constexpr std::size_t kSimCount = 16;
    std::vector<float*> device_ranks(kSimRanks);
    std::vector<std::vector<float>> host_ranks(kSimRanks, std::vector<float>(kSimCount));
    std::vector<matrix_pro::CollectiveBuffer> sim(kSimRanks);
    for (int rank = 0; rank < kSimRanks; ++rank) {
        for (std::size_t i = 0; i < kSimCount; ++i) {
            host_ranks[static_cast<std::size_t>(rank)][i] = static_cast<float>(rank + 1);
        }
        device_ranks[static_cast<std::size_t>(rank)] = matrix_pro::hw::cuda_alloc(0, kSimCount);
        matrix_pro::hw::cuda_copy_h2d(0, device_ranks[static_cast<std::size_t>(rank)],
                                      host_ranks[static_cast<std::size_t>(rank)].data(), kSimCount);
        sim[static_cast<std::size_t>(rank)].data = device_ranks[static_cast<std::size_t>(rank)];
        sim[static_cast<std::size_t>(rank)].count = kSimCount;
        sim[static_cast<std::size_t>(rank)].device = matrix_pro::DeviceId::cuda(0);
    }
    matrix_pro::hw::cuda_sync(0);
    matrix_pro::all_reduce(sim.data(), kSimRanks, matrix_pro::ReduceOp::Sum, matrix_pro::CollectiveAlgorithm::Ring);
    bool sim_ok = true;
    for (int rank = 0; rank < kSimRanks; ++rank) {
        std::vector<float> back(kSimCount);
        matrix_pro::hw::cuda_copy_d2h(0, back.data(), device_ranks[static_cast<std::size_t>(rank)], kSimCount);
        matrix_pro::hw::cuda_sync(0);
        for (float value : back) {
            if (value != 10.0f) {
                sim_ok = false;
            }
        }
        matrix_pro::hw::cuda_free(0, device_ranks[static_cast<std::size_t>(rank)]);
    }
    expect(sim_ok, "simulated multi-gpu ring sum");

    const matrix_pro::MachineTopology topology = matrix_pro::discover_topology();
    expect(topology.logical_cpus >= 1 && topology.numa_nodes >= 1, "topology");

    const int hm = 128;
    const int hn = 64;
    const int hk = 32;
    std::vector<float> ha(static_cast<std::size_t>(hm * hk));
    std::vector<float> hb(static_cast<std::size_t>(hk * hn));
    std::vector<float> hc(static_cast<std::size_t>(hm * hn), 0.0f);
    std::vector<float> href(hc);
    fill_pattern(ha);
    fill_pattern(hb);
    matrix_pro::HeterogeneousScheduler& scheduler = matrix_pro::HeterogeneousScheduler::instance();
    scheduler.calibrate();
    scheduler.gemm(ha.data(), hk, hb.data(), hn, hc.data(), hn, hm, hn, hk, 1.0f, 0.0f);
    matrix_pro::gemm_reference(hm, hn, hk, ha.data(), hk, hb.data(), hn, href.data(), hn, 1.0f, 0.0f);
    const matrix_pro::GemmSplit split = scheduler.last_split();
    std::cout << "  info  hybrid cpu_rows=" << split.cpu_rows << " gpu_rows=" << split.gpu_rows
              << " cpu_gflops=" << split.cpu_gflops << " gpu_gflops=" << split.gpu_gflops << "\n";
    const float hybrid_error = max_abs(hc, href);
    std::cout << "  info  hybrid_max_abs=" << hybrid_error << "\n";
    expect(split.cpu_rows > 0 && split.gpu_rows > 0, "hybrid uses both devices");
    expect(hybrid_error < 1.0e-5f, "hybrid GEMM < 1e-5");
#endif
}

}  // namespace

int main() {
    std::cout << "Nexus Faz 2 donanim testi\n";
    try {
        test_cpu_gemm();
        test_collectives();
        test_hlo_and_pipeline();
        test_cuda_and_hybrid();
    } catch (const std::exception& error) {
        std::cout << "  FAIL  istisna: " << error.what() << "\n";
        return 1;
    }
    if (g_failed != 0) {
        std::cout << g_failed << " kontrol basarisiz\n";
        return 1;
    }
    std::cout << "tum kontroller gecti\n";
    return 0;
}
