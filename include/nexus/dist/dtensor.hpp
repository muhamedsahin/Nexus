#pragma once

/**
 * @file dtensor.hpp
 * @brief FAZ 12.A: DeviceMesh (cok boyutlu cihaz agi, boyut basina alt ProcessGroup) ve DTensor
 *        (yerel parca + yerlesim listesi), yeniden dagitim (redistribute) ve op yayilim kurallari.
 *
 * DeviceMesh(world, shape, names): rank'ler satir-ana (son boyut en hizli) koordinatlara yerlesir.
 * Her boyut icin ayni diger koordinatlari paylasan rank'lerden alt grup kurulur (kollektif: tum
 * rank'ler ayni sirayla kurar). flatten({"dp","cp"}) birden cok boyutu tek grupta birlestirir.
 *
 * Yerlesim (mesh boyutu basina bir tane): Shard(d) — tensor boyutu d mesh boyutu boyunca parcali
 * (torch.chunk kurali: parca = ceil(n / k); son rank'ler kisa ya da bos olabilir), Replicate — tum
 * rank'lerde ayni deger, Partial — gercek deger rank'lerin toplami (Partial(sum)).
 * Kisit: bir tensor boyutu en fazla bir mesh boyutunca parcalanir (ic ice parcalama yok; aksi acik hata).
 *
 * redistribute kurallari (mesh boyutu basina, o boyutun alt grubunda):
 *   Shard(i) -> Replicate  : all-gather (dolgu + paketleme; d != 0 ise boyut i disa alinir)
 *   Partial  -> Replicate  : all-reduce (Sum)
 *   Partial  -> Shard(i)   : reduce-scatter
 *   Shard(i) -> Shard(j)   : all-to-all(v)
 *   Replicate -> Shard(i)  : yerel dilim (iletisim yok)
 *   Replicate -> Partial   : koordinat 0 degeri tutar, digerleri 0
 *   Shard(i) -> Partial    : all-gather + Replicate -> Partial
 * Toplama dogrusal oldugu icin mesh boyutlari birbirinden bagimsiz islenir.
 *
 * Op yayilim kurallari (propagate): matmul, eleman eleman (add/mul), norm (son boyut), softmax(dim),
 * embedding — girislerin hangi yerlesime getirilecegini ve cikis yerlesimini mesh boyutu basina
 * verir. dt_* islemleri bu kurali uygular (gerekirse girisleri yeniden dagitir) ve yerel hesaplar.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/dist/process_group.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace nexus {
namespace dist {

class DeviceMesh {
 public:
  /// Kollektif: world grubunun tum rank'leri ayni shape/names ile cagirir. prod(shape) == world boyu.
  DeviceMesh(GroupPtr world, std::vector<int> shape, std::vector<std::string> names);

  [[nodiscard]] int ndim() const noexcept { return static_cast<int>(shape_.size()); }
  [[nodiscard]] const std::vector<int>& shape() const noexcept { return shape_; }
  [[nodiscard]] const std::vector<std::string>& names() const noexcept { return names_; }
  [[nodiscard]] int size(int dim) const { return shape_.at(static_cast<std::size_t>(dim)); }
  /// Ada gore boyut boyu (yoksa 1).
  [[nodiscard]] int size(const std::string& name) const;
  [[nodiscard]] int coordinate(int dim) const { return coords_.at(static_cast<std::size_t>(dim)); }
  [[nodiscard]] int coordinate(const std::string& name) const;
  [[nodiscard]] const std::vector<int>& coordinates() const noexcept { return coords_; }
  [[nodiscard]] int dim_index(const std::string& name) const noexcept;
  [[nodiscard]] const GroupPtr& group(int dim) const { return groups_.at(static_cast<std::size_t>(dim)); }
  [[nodiscard]] const GroupPtr& group(const std::string& name) const;
  [[nodiscard]] const GroupPtr& world() const noexcept { return world_; }
  [[nodiscard]] int rank() const noexcept { return world_->rank(); }
  [[nodiscard]] int rank_of(const std::vector<int>& coords) const;
  /// Kollektif: verilen boyutlarin birlesimi (diger koordinatlar sabit). Ayni dizi icin onbellekli.
  [[nodiscard]] GroupPtr flatten(const std::vector<std::string>& dims);

 private:
  GroupPtr world_;
  std::vector<int> shape_;
  std::vector<std::string> names_;
  std::vector<int> coords_;
  std::vector<GroupPtr> groups_;
  std::vector<std::pair<std::string, GroupPtr>> flat_;
};

struct Placement {
  enum class Kind : std::uint8_t { Shard, Replicate, Partial };
  Kind kind = Kind::Replicate;
  int dim = 0;  // yalniz Shard

  [[nodiscard]] static Placement shard(int d) { return {Kind::Shard, d}; }
  [[nodiscard]] static Placement replicate() { return {Kind::Replicate, 0}; }
  [[nodiscard]] static Placement partial() { return {Kind::Partial, 0}; }
  [[nodiscard]] bool is_shard() const noexcept { return kind == Kind::Shard; }
  [[nodiscard]] bool is_shard(int d) const noexcept { return kind == Kind::Shard && dim == d; }
  [[nodiscard]] bool is_replicate() const noexcept { return kind == Kind::Replicate; }
  [[nodiscard]] bool is_partial() const noexcept { return kind == Kind::Partial; }
  [[nodiscard]] std::string str() const;
  friend bool operator==(const Placement& a, const Placement& b) noexcept {
    return a.kind == b.kind && (a.kind != Kind::Shard || a.dim == b.dim);
  }
};
using Placements = std::vector<Placement>;
[[nodiscard]] std::string to_string(const Placements& p);

/// torch.chunk kurali: n ogenin k parcaya bolunmesinde parca i'nin [baslangic, boy) araligi.
struct ChunkRange {
  std::int64_t begin = 0;
  std::int64_t length = 0;
};
[[nodiscard]] ChunkRange chunk_range(std::int64_t n, int k, int i) noexcept;

class DTensor {
 public:
  DTensor() = default;
  /// Yerel parcadan (Partial: her rank kendi toplanan payini verir). Sekil kontrol edilir.
  [[nodiscard]] static DTensor from_local(std::vector<float> local, std::vector<std::int64_t> global_shape,
                                          DeviceMesh& mesh, Placements placements);
  /// Tum rank'lerde ayni tam degerden (Partial: koordinat 0 tutar, digerleri 0).
  [[nodiscard]] static DTensor distribute(const std::vector<float>& full, std::vector<std::int64_t> global_shape,
                                          DeviceMesh& mesh, Placements placements);

  [[nodiscard]] const std::vector<float>& local() const noexcept { return local_; }
  [[nodiscard]] std::vector<float>& local() noexcept { return local_; }
  [[nodiscard]] const std::vector<std::int64_t>& shape() const noexcept { return shape_; }
  [[nodiscard]] std::vector<std::int64_t> local_shape() const;
  [[nodiscard]] const Placements& placements() const noexcept { return placements_; }
  [[nodiscard]] DeviceMesh& mesh() const { return *mesh_; }

  [[nodiscard]] DTensor redistribute(const Placements& target) const;
  /// Kollektif: tam deger (tum mesh boyutlari Replicate).
  [[nodiscard]] std::vector<float> full_tensor() const;

  /// Yerlesim + sekil icin yerel sekil ve yerel blogun her tensor boyutundaki baslangici.
  [[nodiscard]] static std::vector<std::int64_t> local_shape_of(const std::vector<std::int64_t>& global,
                                                                const DeviceMesh& mesh, const Placements& p);
  [[nodiscard]] static std::vector<std::int64_t> local_offset_of(const std::vector<std::int64_t>& global,
                                                                 const DeviceMesh& mesh, const Placements& p);
  /// Yerlesim gecerli mi (boy, boyut indeksleri, ic ice parcalama yok). Gecersizse DistError.
  static void validate(const std::vector<std::int64_t>& global, const DeviceMesh& mesh, const Placements& p);

 private:
  std::vector<float> local_;
  std::vector<std::int64_t> shape_;
  Placements placements_;
  DeviceMesh* mesh_ = nullptr;
};

// ---- Op yayilim kurallari ----------------------------------------------------------------

enum class DOp : std::uint8_t { Matmul, Add, Mul, Norm, Softmax, Embedding };
[[nodiscard]] const char* to_string(DOp op) noexcept;

struct ShardingDecision {
  std::vector<Placements> inputs;  // girislerin getirilecegi yerlesim
  Placements output;
  std::vector<std::string> rules;  // mesh boyutu basina uygulanan kural (rapor/test)
};

/// inputs: girislerin su anki yerlesimleri. Matmul: A [m,k], B [k,n]. Add/Mul: ayni sekil.
/// Norm: [N, C] son boyut uzerinde (agirlik Replicate). Softmax: dim boyutu. Embedding: tablo [V, C],
/// kimlikler [N] (kimlik yerlesimi Shard(0) ya da Replicate).
[[nodiscard]] ShardingDecision propagate(DOp op, const std::vector<Placements>& inputs, int dim = -1);

/// Kurali uygular ve yerel hesaplar (girisler gerekirse yeniden dagitilir).
[[nodiscard]] DTensor dt_matmul(const DTensor& a, const DTensor& b);
[[nodiscard]] DTensor dt_add(const DTensor& a, const DTensor& b);
[[nodiscard]] DTensor dt_mul(const DTensor& a, const DTensor& b);
/// LayerNorm (son boyut). gamma/beta Replicate tam [C] vektorleri.
[[nodiscard]] DTensor dt_layer_norm(const DTensor& x, const std::vector<float>& gamma, const std::vector<float>& beta,
                                    float eps);
[[nodiscard]] DTensor dt_softmax(const DTensor& x, int dim);
/// ids: tam kimlik listesi (tum rank'lerde ayni; Replicate). Tablo Shard(0) (sozluk) -> cikis Partial.
[[nodiscard]] DTensor dt_embedding(const DTensor& table, const std::vector<std::int32_t>& ids);

}  // namespace dist
}  // namespace nexus
