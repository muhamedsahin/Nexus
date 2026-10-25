#pragma once

/**
 * @file topology.hpp
 * @brief FAZ 10.5: kume topolojisi — MatrixFlash discover_topology (NUMA, CUDA P2P / NVLink matrisi)
 *        + dugum ici / dugumler arasi ayrimi (LOCAL_WORLD_SIZE) ve all-reduce algoritmasi onerisi.
 *
 * Oneri: tek dugum -> "ring"; birden cok dugum -> "hierarchical" (DdpOptions::ranks_per_node =
 * LOCAL_WORLD_SIZE). P2P/NVLink matrisi bilgi amaclidir: bu derlemede GPU'lar arasi dogrudan kopya
 * yolu yok (tek GPU, WDDM), aktarim host uzerinden.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include <string>
#include <vector>

namespace nexus {
namespace dist {

struct ClusterTopology {
  int world_size = 1;
  int ranks_per_node = 1;
  int nodes = 1;
  int node_rank = 0;
  int local_rank = 0;
  int logical_cpus = 1;
  int numa_nodes = 1;
  int cuda_devices = 0;
  std::vector<std::vector<int>> peer_access;  // [i][j] = 1: cudaDeviceCanAccessPeer
  std::vector<std::vector<int>> nvlink;       // [i][j] = 1: NVLink sinifi baglanti
  std::string algorithm = "ring";
  [[nodiscard]] std::string describe() const;
};

/// world/rank/ranks_per_node verilmezse (<= 0) ortamdan: WORLD_SIZE, RANK, LOCAL_WORLD_SIZE.
[[nodiscard]] ClusterTopology discover_cluster(int world_size = 0, int rank = -1, int ranks_per_node = 0);

}  // namespace dist
}  // namespace nexus
