/**
 * @file topology.cpp
 * @brief FAZ 10.5: kume topolojisi (MatrixFlash donanim kesfi + rank yerlesimi).
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/dist/topology.hpp"

#include "nexus/dist/process_group.hpp"

#if defined(NEXUS_HAS_MATRIX_PRO)
#include "matrix_pro/distributed/topology.hpp"
#endif

#include <sstream>
#include <thread>

namespace nexus {
namespace dist {

ClusterTopology discover_cluster(int world_size, int rank, int ranks_per_node) {
  ClusterTopology t;
  t.world_size = world_size > 0 ? world_size : env_int("WORLD_SIZE", 1);
  const int r = rank >= 0 ? rank : env_int("RANK", 0);
  t.ranks_per_node = ranks_per_node > 0 ? ranks_per_node : env_int("LOCAL_WORLD_SIZE", t.world_size);
  if (t.ranks_per_node < 1 || t.world_size % t.ranks_per_node != 0) {
    throw DistError("Nexus dist: LOCAL_WORLD_SIZE dunya boyunu bolmeli");
  }
  t.nodes = t.world_size / t.ranks_per_node;
  t.node_rank = r / t.ranks_per_node;
  t.local_rank = r % t.ranks_per_node;
  t.algorithm = t.nodes > 1 ? "hierarchical" : "ring";
#if defined(NEXUS_HAS_MATRIX_PRO)
  const matrix_pro::MachineTopology m = matrix_pro::discover_topology();
  t.logical_cpus = m.logical_cpus;
  t.numa_nodes = m.numa_nodes;
  t.cuda_devices = m.cuda_devices;
  t.peer_access.assign(static_cast<std::size_t>(m.cuda_devices), std::vector<int>(static_cast<std::size_t>(m.cuda_devices), 0));
  t.nvlink = t.peer_access;
  for (const matrix_pro::PeerLink& l : m.links) {
    if (l.device_a < m.cuda_devices && l.device_b < m.cuda_devices) {
      t.peer_access[static_cast<std::size_t>(l.device_a)][static_cast<std::size_t>(l.device_b)] = l.peer_access ? 1 : 0;
      t.nvlink[static_cast<std::size_t>(l.device_a)][static_cast<std::size_t>(l.device_b)] = l.nvlink_class ? 1 : 0;
    }
  }
#else
  t.logical_cpus = static_cast<int>(std::thread::hardware_concurrency());
#endif
  return t;
}

std::string ClusterTopology::describe() const {
  std::ostringstream s;
  s << "dunya " << world_size << " = " << nodes << " dugum x " << ranks_per_node << " rank (bu rank: dugum " << node_rank
    << ", yerel " << local_rank << "); " << logical_cpus << " mantiksal CPU, " << numa_nodes << " NUMA, " << cuda_devices
    << " CUDA aygiti";
  if (cuda_devices > 1) {
    s << "; P2P matrisi:";
    for (std::size_t i = 0; i < peer_access.size(); ++i) {
      s << " [";
      for (std::size_t j = 0; j < peer_access[i].size(); ++j) s << (i == j ? '-' : (nvlink[i][j] ? 'N' : (peer_access[i][j] ? 'P' : '.')));
      s << "]";
    }
  }
  s << "; all-reduce: " << algorithm;
  return s.str();
}

}  // namespace dist
}  // namespace nexus
