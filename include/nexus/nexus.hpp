#pragma once

/**
 * ============================================================================
 * Nexus Deep Learning & Reinforcement Learning Framework
 * Master Umbrella Header (include/nexus/nexus.hpp)
 * ============================================================================
 * Bursa Teknik Üniversitesi Bilgisayar Mühendisliği
 * Geliştirici: Muhammed Fatih Şahin
 * ============================================================================
 */

// 1. Hardware-Accelerated Matrix, Tensor & Autograd Engine (MatrixFlash-Pro)
#if __has_include("matrix_pro/matrix_pro.hpp")
#include "matrix_pro/matrix_pro.hpp"
#endif

// 2. High-Throughput Zero-Copy Data Loading Engine (NexusData)
#if __has_include("nexusdata/nexusdata.hpp")
#include "nexusdata/nexusdata.hpp"
#endif

// 3. Comprehensive Loss Functions Suite (NexusLoss)
#if __has_include("nexusloss/nexusloss.hpp")
#include "nexusloss/nexusloss.hpp"
#endif

// 4. Ultra-Fast SIMD & GPU Optimization Algorithms (NexusOptim)
#if __has_include("nexus_optim/nexus_optim.hpp")
#include "nexus_optim/nexus_optim.hpp"
#endif

// 5. Neural Network Layers, Containers & Explicit Backward (NexusModel)
#if __has_include("nexus_model/nexus_model.hpp")
#include "nexus_model/nexus_model.hpp"
#endif

// 6. Training Loop Orchestrator, Callbacks, Mixed Precision & DDP (NexusTrain)
#if __has_include("nexus_train/nexus_train.hpp")
#include "nexus_train/nexus_train.hpp"
#endif

// 7. Sifir kopyali birlesik tensör koprusu ve tahsisatsiz geri yayilim.
#include "nexus/core/tensor_bridge.hpp"
#include "nexus/loss/backward_into.hpp"

// 8. Kullanici aktivasyonu, kaybi, katmani, optimizatoru, metrigi ve callback'i.
#include "nexus/custom/custom.hpp"

// 9. Pekistirmeli ve cok ajanli ogrenme.
#include "nexus/rl/rl.hpp"
#include "nexus/rl/buffers.hpp"
#include "nexus/rl/agents.hpp"

// 10. Akici yuksek seviye API: model, kayip, optimizer, veri ve egitmen.
#include "nexus/core/config.hpp"
#include "nexus/nn/modules.hpp"
#include "nexus/nn/activations.hpp"
#include "nexus/nn/functional.hpp"
#include "nexus/loss/losses.hpp"
#include "nexus/optim/optimizers.hpp"
#include "nexus/optim/schedulers.hpp"
#include "nexus/data/datasets.hpp"
#include "nexus/metrics/metrics.hpp"
#include "nexus/train/callbacks.hpp"
#include "nexus/train/distributed.hpp"
#include "nexus/train/trainer.hpp"
