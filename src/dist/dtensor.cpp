/**
 * @file dtensor.cpp
 * @brief FAZ 12.A: DeviceMesh, DTensor redistribute, op yayilim kurallari ve dt_* islemleri.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/dist/dtensor.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>

namespace nexus {
namespace dist {

// ==========================================================================================
// DeviceMesh
// ==========================================================================================

DeviceMesh::DeviceMesh(GroupPtr world, std::vector<int> shape, std::vector<std::string> names)
    : world_(std::move(world)), shape_(std::move(shape)), names_(std::move(names)) {
  if (!world_) throw DistError("Nexus DeviceMesh: world grubu bos");
  if (shape_.empty() || shape_.size() != names_.size()) {
    throw DistError("Nexus DeviceMesh: shape ve names ayni boyda ve bos olmamali");
  }
  int total = 1;
  for (int s : shape_) {
    if (s < 1) throw DistError("Nexus DeviceMesh: boyut boyu >= 1 olmali");
    total *= s;
  }
  if (total != world_->size()) {
    throw DistError("Nexus DeviceMesh: prod(shape) = " + std::to_string(total) + " != world boyu " +
                    std::to_string(world_->size()));
  }
  for (std::size_t i = 0; i < names_.size(); ++i) {
    for (std::size_t j = i + 1; j < names_.size(); ++j) {
      if (names_[i] == names_[j]) throw DistError("Nexus DeviceMesh: tekrar eden boyut adi " + names_[i]);
    }
  }
  coords_.resize(shape_.size());
  int rem = world_->rank();
  for (int d = ndim() - 1; d >= 0; --d) {
    coords_[static_cast<std::size_t>(d)] = rem % shape_[static_cast<std::size_t>(d)];
    rem /= shape_[static_cast<std::size_t>(d)];
  }
  groups_.resize(shape_.size());
  for (int d = 0; d < ndim(); ++d) {
    // Boyut d disindaki koordinatlarin her birlesimi icin bir grup (rank sirasiyla; kollektif).
    std::vector<int> other = shape_;
    other[static_cast<std::size_t>(d)] = 1;
    int combos = 1;
    for (int s : other) combos *= s;
    for (int c = 0; c < combos; ++c) {
      std::vector<int> co(shape_.size(), 0);
      int r = c;
      for (int e = ndim() - 1; e >= 0; --e) {
        co[static_cast<std::size_t>(e)] = r % other[static_cast<std::size_t>(e)];
        r /= other[static_cast<std::size_t>(e)];
      }
      std::vector<int> ranks;
      for (int k = 0; k < shape_[static_cast<std::size_t>(d)]; ++k) {
        co[static_cast<std::size_t>(d)] = k;
        ranks.push_back(rank_of(co));
      }
      GroupPtr g = world_->new_group(ranks);
      if (g) groups_[static_cast<std::size_t>(d)] = std::move(g);
    }
  }
}

int DeviceMesh::dim_index(const std::string& name) const noexcept {
  for (std::size_t i = 0; i < names_.size(); ++i) {
    if (names_[i] == name) return static_cast<int>(i);
  }
  return -1;
}

int DeviceMesh::size(const std::string& name) const {
  const int d = dim_index(name);
  return d < 0 ? 1 : size(d);
}

int DeviceMesh::coordinate(const std::string& name) const {
  const int d = dim_index(name);
  return d < 0 ? 0 : coordinate(d);
}

const GroupPtr& DeviceMesh::group(const std::string& name) const {
  const int d = dim_index(name);
  if (d < 0) throw DistError("Nexus DeviceMesh: boyut yok: " + name);
  return group(d);
}

int DeviceMesh::rank_of(const std::vector<int>& coords) const {
  int r = 0;
  for (std::size_t d = 0; d < shape_.size(); ++d) {
    r = r * shape_[d] + coords[d];
  }
  return r;
}

GroupPtr DeviceMesh::flatten(const std::vector<std::string>& dims) {
  std::string key;
  std::vector<bool> in(shape_.size(), false);
  for (const std::string& n : dims) {
    const int d = dim_index(n);
    if (d < 0) throw DistError("Nexus DeviceMesh::flatten: boyut yok: " + n);
    in[static_cast<std::size_t>(d)] = true;
  }
  for (std::size_t d = 0; d < shape_.size(); ++d) {
    if (in[d]) key += names_[d] + ",";
  }
  for (const auto& kv : flat_) {
    if (kv.first == key) return kv.second;
  }
  std::vector<int> other = shape_;
  std::vector<int> inner = shape_;
  for (std::size_t d = 0; d < shape_.size(); ++d) {
    if (in[d]) {
      other[d] = 1;
    } else {
      inner[d] = 1;
    }
  }
  const int combos = std::accumulate(other.begin(), other.end(), 1, std::multiplies<int>());
  const int members = std::accumulate(inner.begin(), inner.end(), 1, std::multiplies<int>());
  GroupPtr mine;
  for (int c = 0; c < combos; ++c) {
    std::vector<int> base(shape_.size(), 0);
    int r = c;
    for (int e = ndim() - 1; e >= 0; --e) {
      base[static_cast<std::size_t>(e)] = r % other[static_cast<std::size_t>(e)];
      r /= other[static_cast<std::size_t>(e)];
    }
    std::vector<int> ranks;
    for (int m = 0; m < members; ++m) {
      std::vector<int> co = base;
      int q = m;
      for (int e = ndim() - 1; e >= 0; --e) {
        if (in[static_cast<std::size_t>(e)]) {
          co[static_cast<std::size_t>(e)] = q % inner[static_cast<std::size_t>(e)];
          q /= inner[static_cast<std::size_t>(e)];
        }
      }
      ranks.push_back(rank_of(co));
    }
    GroupPtr g = world_->new_group(ranks);
    if (g) mine = std::move(g);
  }
  flat_.emplace_back(key, mine);
  return mine;
}

// ==========================================================================================
// Yerlesim yardimcilari
// ==========================================================================================

std::string Placement::str() const {
  switch (kind) {
    case Kind::Shard:
      return "S(" + std::to_string(dim) + ")";
    case Kind::Replicate:
      return "R";
    case Kind::Partial:
      return "P";
  }
  return "?";
}

std::string to_string(const Placements& p) {
  std::string s = "[";
  for (std::size_t i = 0; i < p.size(); ++i) {
    if (i) s += ",";
    s += p[i].str();
  }
  return s + "]";
}

ChunkRange chunk_range(std::int64_t n, int k, int i) noexcept {
  const std::int64_t c = (n + k - 1) / k;
  const std::int64_t b = std::min<std::int64_t>(n, static_cast<std::int64_t>(i) * c);
  return {b, std::clamp<std::int64_t>(n - static_cast<std::int64_t>(i) * c, 0, c)};
}

namespace {

std::int64_t numel_of(const std::vector<std::int64_t>& s) {
  std::int64_t n = 1;
  for (std::int64_t v : s) n *= v;
  return n;
}

/// [pre, n, post] gorunumu (tensor boyutu dim icin).
struct Split3 {
  std::int64_t pre = 1, n = 1, post = 1;
};
Split3 split3(const std::vector<std::int64_t>& s, int dim) {
  Split3 r;
  for (int i = 0; i < dim; ++i) r.pre *= s[static_cast<std::size_t>(i)];
  r.n = s[static_cast<std::size_t>(dim)];
  for (std::size_t i = static_cast<std::size_t>(dim) + 1; i < s.size(); ++i) r.post *= s[i];
  return r;
}

/// src [pre, sn, post]'tan boyut ekseninde [sb, sb + len) -> dst [pre, dn, post] [db, db + len).
void copy_rows(const float* src, std::int64_t sn, std::int64_t sb, float* dst, std::int64_t dn, std::int64_t db,
               std::int64_t len, std::int64_t pre, std::int64_t post) {
  if (len <= 0) return;
  for (std::int64_t p = 0; p < pre; ++p) {
    std::memcpy(dst + (p * dn + db) * post, src + (p * sn + sb) * post, static_cast<std::size_t>(len * post) * sizeof(float));
  }
}

/// Genel N boyutlu alt blok kopyasi: src[off_s + idx] -> dst[off_d + idx], idx < extent.
void copy_block(const float* src, const std::vector<std::int64_t>& ss, const std::vector<std::int64_t>& so, float* dst,
                const std::vector<std::int64_t>& ds, const std::vector<std::int64_t>& dofs,
                const std::vector<std::int64_t>& ext) {
  const std::size_t nd = ss.size();
  if (numel_of(ext) == 0) return;
  std::vector<std::int64_t> idx(nd, 0);
  const std::int64_t last = ext[nd - 1];
  while (true) {
    std::int64_t si = 0, di = 0;
    for (std::size_t d = 0; d < nd; ++d) {
      si = si * ss[d] + so[d] + idx[d];
      di = di * ds[d] + dofs[d] + idx[d];
    }
    std::memcpy(dst + di, src + si, static_cast<std::size_t>(last) * sizeof(float));
    // son boyut haric coklu indeksi arttir
    int d = static_cast<int>(nd) - 2;
    while (d >= 0) {
      if (++idx[static_cast<std::size_t>(d)] < ext[static_cast<std::size_t>(d)]) break;
      idx[static_cast<std::size_t>(d)] = 0;
      --d;
    }
    if (d < 0) return;
  }
}

bool placements_valid(const std::vector<std::int64_t>& global, const Placements& p) {
  std::vector<int> seen(global.size(), 0);
  for (const Placement& x : p) {
    if (x.is_shard()) {
      if (x.dim < 0 || x.dim >= static_cast<int>(global.size())) return false;
      if (seen[static_cast<std::size_t>(x.dim)]++) return false;
    }
  }
  return true;
}

}  // namespace

void DTensor::validate(const std::vector<std::int64_t>& global, const DeviceMesh& mesh, const Placements& p) {
  if (static_cast<int>(p.size()) != mesh.ndim()) {
    throw DistError("Nexus DTensor: yerlesim sayisi mesh boyut sayisina esit olmali (" + to_string(p) + ")");
  }
  for (const Placement& x : p) {
    if (x.is_shard() && (x.dim < 0 || x.dim >= static_cast<int>(global.size()))) {
      throw DistError("Nexus DTensor: Shard boyutu tensor boyut sayisi disinda: " + to_string(p));
    }
  }
  if (!placements_valid(global, p)) {
    throw DistError("Nexus DTensor: bir tensor boyutu birden cok mesh boyutunca parcalanamaz (ic ice parcalama yok): " +
                    to_string(p));
  }
}

std::vector<std::int64_t> DTensor::local_shape_of(const std::vector<std::int64_t>& global, const DeviceMesh& mesh,
                                                  const Placements& p) {
  std::vector<std::int64_t> s = global;
  for (int d = 0; d < mesh.ndim(); ++d) {
    const Placement& x = p[static_cast<std::size_t>(d)];
    if (x.is_shard()) {
      s[static_cast<std::size_t>(x.dim)] =
          chunk_range(global[static_cast<std::size_t>(x.dim)], mesh.size(d), mesh.coordinate(d)).length;
    }
  }
  return s;
}

std::vector<std::int64_t> DTensor::local_offset_of(const std::vector<std::int64_t>& global, const DeviceMesh& mesh,
                                                   const Placements& p) {
  std::vector<std::int64_t> o(global.size(), 0);
  for (int d = 0; d < mesh.ndim(); ++d) {
    const Placement& x = p[static_cast<std::size_t>(d)];
    if (x.is_shard()) {
      o[static_cast<std::size_t>(x.dim)] =
          chunk_range(global[static_cast<std::size_t>(x.dim)], mesh.size(d), mesh.coordinate(d)).begin;
    }
  }
  return o;
}

std::vector<std::int64_t> DTensor::local_shape() const { return local_shape_of(shape_, *mesh_, placements_); }

DTensor DTensor::from_local(std::vector<float> local, std::vector<std::int64_t> global_shape, DeviceMesh& mesh,
                            Placements placements) {
  validate(global_shape, mesh, placements);
  const std::int64_t want = numel_of(local_shape_of(global_shape, mesh, placements));
  if (static_cast<std::int64_t>(local.size()) != want) {
    throw DistError("Nexus DTensor::from_local: yerel boy " + std::to_string(local.size()) + ", beklenen " +
                    std::to_string(want));
  }
  DTensor t;
  t.local_ = std::move(local);
  t.shape_ = std::move(global_shape);
  t.placements_ = std::move(placements);
  t.mesh_ = &mesh;
  return t;
}

DTensor DTensor::distribute(const std::vector<float>& full, std::vector<std::int64_t> global_shape, DeviceMesh& mesh,
                            Placements placements) {
  validate(global_shape, mesh, placements);
  if (static_cast<std::int64_t>(full.size()) != numel_of(global_shape)) {
    throw DistError("Nexus DTensor::distribute: tam deger boyu sekille uyusmuyor");
  }
  const std::vector<std::int64_t> ls = local_shape_of(global_shape, mesh, placements);
  const std::vector<std::int64_t> lo = local_offset_of(global_shape, mesh, placements);
  std::vector<float> local(static_cast<std::size_t>(numel_of(ls)));
  copy_block(full.data(), global_shape, lo, local.data(), ls, std::vector<std::int64_t>(ls.size(), 0), ls);
  bool zero = false;
  for (int d = 0; d < mesh.ndim(); ++d) {
    if (placements[static_cast<std::size_t>(d)].is_partial() && mesh.coordinate(d) != 0) zero = true;
  }
  if (zero) std::fill(local.begin(), local.end(), 0.0f);
  return from_local(std::move(local), std::move(global_shape), mesh, std::move(placements));
}

namespace {

/// Tek mesh boyutunda bir gecis. t'nin yerlesimi ve yereli guncellenir.
void transition(std::vector<float>& local, const std::vector<std::int64_t>& global, DeviceMesh& mesh, Placements& cur,
                int d, const Placement& to) {
  const Placement from = cur[static_cast<std::size_t>(d)];
  ProcessGroup& g = *mesh.group(d);
  const int k = mesh.size(d);
  const int r = mesh.coordinate(d);
  const std::vector<std::int64_t> ls = DTensor::local_shape_of(global, mesh, cur);
  Placements next = cur;
  next[static_cast<std::size_t>(d)] = to;

  if (from.is_shard() && to.is_replicate()) {
    const int i = from.dim;
    const Split3 s = split3(ls, i);
    const std::int64_t n = global[static_cast<std::size_t>(i)];
    const std::int64_t c = (n + k - 1) / k;
    std::vector<float> pack(static_cast<std::size_t>(s.pre * c * s.post), 0.0f);
    copy_rows(local.data(), s.n, 0, pack.data(), c, 0, s.n, s.pre, s.post);
    std::vector<float> all(pack.size() * static_cast<std::size_t>(k));
    g.all_gather(pack.data(), all.data(), pack.size(), DType::F32)->wait();
    std::vector<float> out(static_cast<std::size_t>(s.pre * n * s.post));
    for (int j = 0; j < k; ++j) {
      const ChunkRange cr = chunk_range(n, k, j);
      copy_rows(all.data() + static_cast<std::size_t>(j) * pack.size(), c, 0, out.data(), n, cr.begin, cr.length, s.pre,
                s.post);
    }
    local = std::move(out);
  } else if (from.is_replicate() && to.is_shard()) {
    const int i = to.dim;
    const Split3 s = split3(ls, i);
    const ChunkRange cr = chunk_range(s.n, k, r);
    std::vector<float> out(static_cast<std::size_t>(s.pre * cr.length * s.post));
    copy_rows(local.data(), s.n, cr.begin, out.data(), cr.length, 0, cr.length, s.pre, s.post);
    local = std::move(out);
  } else if (from.is_partial() && to.is_replicate()) {
    if (!local.empty()) g.all_reduce(local.data(), local.size(), DType::F32, ReduceOp::Sum)->wait();
  } else if (from.is_partial() && to.is_shard()) {
    const int i = to.dim;
    const Split3 s = split3(ls, i);
    const std::int64_t c = (s.n + k - 1) / k;
    const std::size_t chunk = static_cast<std::size_t>(s.pre * c * s.post);
    std::vector<float> pack(chunk * static_cast<std::size_t>(k), 0.0f);
    for (int j = 0; j < k; ++j) {
      const ChunkRange cr = chunk_range(s.n, k, j);
      copy_rows(local.data(), s.n, cr.begin, pack.data() + static_cast<std::size_t>(j) * chunk, c, 0, cr.length, s.pre,
                s.post);
    }
    std::vector<float> mine(chunk);
    g.reduce_scatter(pack.data(), mine.data(), chunk, DType::F32, ReduceOp::Sum)->wait();
    const ChunkRange my = chunk_range(s.n, k, r);
    std::vector<float> out(static_cast<std::size_t>(s.pre * my.length * s.post));
    copy_rows(mine.data(), c, 0, out.data(), my.length, 0, my.length, s.pre, s.post);
    local = std::move(out);
  } else if (from.is_shard() && to.is_shard()) {
    // all-to-all: boyut j boyunca bolup gonder; gelenleri boyut i boyunca birlestir.
    const int i = from.dim;
    const int j = to.dim;
    const std::vector<std::int64_t> zero(ls.size(), 0);
    std::vector<std::size_t> send_counts(static_cast<std::size_t>(k)), recv_counts(static_cast<std::size_t>(k));
    std::vector<float> send;
    for (int t = 0; t < k; ++t) {
      const ChunkRange cr = chunk_range(ls[static_cast<std::size_t>(j)], k, t);
      std::vector<std::int64_t> ext = ls;
      ext[static_cast<std::size_t>(j)] = cr.length;
      std::vector<std::int64_t> so = zero;
      so[static_cast<std::size_t>(j)] = cr.begin;
      const std::size_t base = send.size();
      send.resize(base + static_cast<std::size_t>(numel_of(ext)));
      copy_block(local.data(), ls, so, send.data() + base, ext, zero, ext);
      send_counts[static_cast<std::size_t>(t)] = static_cast<std::size_t>(numel_of(ext));
    }
    const std::int64_t ni = global[static_cast<std::size_t>(i)];
    const ChunkRange myj = chunk_range(ls[static_cast<std::size_t>(j)], k, r);
    for (int s = 0; s < k; ++s) {
      std::vector<std::int64_t> ext = ls;
      ext[static_cast<std::size_t>(i)] = chunk_range(ni, k, s).length;
      ext[static_cast<std::size_t>(j)] = myj.length;
      recv_counts[static_cast<std::size_t>(s)] = static_cast<std::size_t>(numel_of(ext));
    }
    std::vector<float> recv(std::accumulate(recv_counts.begin(), recv_counts.end(), std::size_t{0}));
    g.all_to_all_v(send.data(), send_counts, recv.data(), recv_counts, DType::F32)->wait();
    std::vector<std::int64_t> os = ls;
    os[static_cast<std::size_t>(i)] = ni;
    os[static_cast<std::size_t>(j)] = myj.length;
    std::vector<float> out(static_cast<std::size_t>(numel_of(os)));
    std::size_t off = 0;
    for (int s = 0; s < k; ++s) {
      std::vector<std::int64_t> ext = os;
      const ChunkRange ci = chunk_range(ni, k, s);
      ext[static_cast<std::size_t>(i)] = ci.length;
      std::vector<std::int64_t> dofs = zero;
      dofs[static_cast<std::size_t>(i)] = ci.begin;
      copy_block(recv.data() + off, ext, zero, out.data(), os, dofs, ext);
      off += recv_counts[static_cast<std::size_t>(s)];
    }
    local = std::move(out);
  } else if (from.is_replicate() && to.is_partial()) {
    if (r != 0) std::fill(local.begin(), local.end(), 0.0f);
  } else if (from.is_shard() && to.is_partial()) {
    transition(local, global, mesh, cur, d, Placement::replicate());
    transition(local, global, mesh, cur, d, Placement::partial());
    return;
  } else {
    throw DistError("Nexus DTensor: desteklenmeyen gecis " + from.str() + " -> " + to.str());
  }
  cur = next;
}

}  // namespace

DTensor DTensor::redistribute(const Placements& target) const {
  validate(shape_, *mesh_, target);
  DTensor out = *this;
  std::vector<bool> done(target.size(), false);
  for (std::size_t d = 0; d < target.size(); ++d) done[d] = out.placements_[d] == target[d];
  // Gecerli (ic ice parcalama uretmeyen) gecisleri yap; ilerleme yoksa bir parcayi once Replicate'e topla.
  for (int guard = 0; guard < 4 * static_cast<int>(target.size()) + 4; ++guard) {
    bool all = true;
    bool progress = false;
    for (std::size_t d = 0; d < target.size(); ++d) {
      if (done[d]) continue;
      all = false;
      Placements trial = out.placements_;
      trial[d] = target[d];
      if (placements_valid(shape_, trial)) {
        transition(out.local_, shape_, *mesh_, out.placements_, static_cast<int>(d), target[d]);
        done[d] = true;
        progress = true;
      }
    }
    if (all) return out;
    if (!progress) {
      for (std::size_t d = 0; d < target.size(); ++d) {
        if (!done[d] && out.placements_[d].is_shard()) {
          transition(out.local_, shape_, *mesh_, out.placements_, static_cast<int>(d), Placement::replicate());
          break;
        }
      }
    }
  }
  for (std::size_t d = 0; d < target.size(); ++d) {
    if (!(out.placements_[d] == target[d])) throw DistError("Nexus DTensor::redistribute: hedefe ulasilamadi");
  }
  return out;
}

std::vector<float> DTensor::full_tensor() const {
  return redistribute(Placements(placements_.size(), Placement::replicate())).local_;
}

// ==========================================================================================
// Op yayilim kurallari
// ==========================================================================================

const char* to_string(DOp op) noexcept {
  switch (op) {
    case DOp::Matmul:
      return "matmul";
    case DOp::Add:
      return "add";
    case DOp::Mul:
      return "mul";
    case DOp::Norm:
      return "norm";
    case DOp::Softmax:
      return "softmax";
    case DOp::Embedding:
      return "embedding";
  }
  return "?";
}

namespace {

using P = Placement;

struct Rule1 {
  std::vector<Placement> in;
  Placement out;
  std::string name;
};

Rule1 rule_1d(DOp op, const std::vector<Placement>& in, int dim) {
  const P R = P::replicate();
  const P Pa = P::partial();
  switch (op) {
    case DOp::Matmul: {
      const P a = in[0], b = in[1];
      if (a.is_shard(0) && b.is_replicate()) return {{a, b}, P::shard(0), "S0 x R -> S0 (satir)"};
      if (a.is_replicate() && b.is_shard(1)) return {{a, b}, P::shard(1), "R x S1 -> S1 (sutun)"};
      if (a.is_shard(1) && b.is_shard(0)) return {{a, b}, Pa, "S1 x S0 -> P (daralan eksen)"};
      if (a.is_replicate() && b.is_replicate()) return {{a, b}, R, "R x R -> R"};
      if (a.is_partial() && b.is_replicate()) return {{a, b}, Pa, "P x R -> P"};
      if (a.is_replicate() && b.is_partial()) return {{a, b}, Pa, "R x P -> P"};
      if (a.is_shard(0)) return {{a, R}, P::shard(0), "S0 x ? : B -> R; S0"};
      if (a.is_shard(1)) return {{a, P::shard(0)}, Pa, "S1 x ? : B -> S0; P"};
      if (a.is_replicate() && b.is_shard(0)) return {{P::shard(1), b}, Pa, "R x S0 : A -> S1; P"};
      if (a.is_partial() && b.is_shard(0)) return {{P::shard(1), b}, Pa, "P x S0 : A -> S1 (RS); P"};
      if (a.is_partial() && b.is_shard(1)) return {{R, b}, P::shard(1), "P x S1 : A -> R; S1"};
      return {{R, b}, Pa, "P x P : A -> R; P"};
    }
    case DOp::Add: {
      const P a = in[0], b = in[1];
      if (a == b) return {{a, b}, a, "ayni yerlesim korunur"};
      if (a.is_partial() && b.is_replicate()) return {{R, b}, R, "P + R : A -> R; R"};
      if (a.is_partial()) return {{b, b}, b, "P + S : A -> S (RS)"};
      return {{a, a}, a, "B -> A'nin yerlesimi"};
    }
    case DOp::Mul: {
      const P a = in[0], b = in[1];
      if (a == b && !a.is_partial()) return {{a, b}, a, "ayni yerlesim korunur"};
      if (a.is_partial() && b.is_replicate()) return {{a, b}, Pa, "P * R -> P (dogrusal)"};
      if (a.is_replicate() && b.is_partial()) return {{a, b}, Pa, "R * P -> P (dogrusal)"};
      if (a.is_partial()) return {{a, R}, Pa, "P * ? : B -> R; P"};
      return {{a, a}, a, "B -> A'nin yerlesimi"};
    }
    case DOp::Norm: {
      const P x = in[0];
      if (x.is_shard(0)) return {{x}, x, "S0 korunur (satir basina norm)"};
      if (x.is_replicate()) return {{x}, x, "R"};
      return {{R}, R, x.is_partial() ? "P -> R (norm dogrusal degil)" : "S(son) -> R (norm ekseni)"};
    }
    case DOp::Softmax: {
      const P x = in[0];
      if (x.is_shard() && x.dim != dim) return {{x}, x, "S(diger) korunur"};
      if (x.is_replicate()) return {{x}, x, "R"};
      return {{R}, R, x.is_partial() ? "P -> R" : "S(softmax ekseni) -> R"};
    }
    case DOp::Embedding: {
      const P t = in[0];
      const P ids = in.size() > 1 ? in[1] : R;
      if (t.is_shard(0)) return {{t, ids}, Pa, "tablo S0 (sozluk) -> P (maskeli arama)"};
      if (t.is_shard(1)) return {{t, ids}, P::shard(1), "tablo S1 -> S1"};
      if (t.is_replicate()) return {{t, ids}, R, "R -> R"};
      return {{R, ids}, R, "tablo P -> R"};
    }
  }
  throw DistError("Nexus propagate: bilinmeyen op");
}

}  // namespace

ShardingDecision propagate(DOp op, const std::vector<Placements>& inputs, int dim) {
  if (inputs.empty()) throw DistError("Nexus propagate: giris yok");
  const std::size_t nd = inputs[0].size();
  ShardingDecision dec;
  dec.inputs.assign(inputs.size(), Placements(nd));
  dec.output.resize(nd);
  dec.rules.resize(nd);
  for (std::size_t d = 0; d < nd; ++d) {
    std::vector<Placement> in;
    for (const Placements& p : inputs) in.push_back(p.at(d));
    Rule1 r = rule_1d(op, in, dim);
    for (std::size_t i = 0; i < inputs.size(); ++i) dec.inputs[i][d] = r.in[i];
    dec.output[d] = r.out;
    dec.rules[d] = r.name;
  }
  // Ic ice parcalama ureten mesh boyutlari R x R -> R'ye (ya da girisin R'sine) duser.
  auto nested = [](const Placements& p) {
    std::vector<int> seen;
    for (const Placement& x : p) {
      if (x.is_shard()) {
        if (std::find(seen.begin(), seen.end(), x.dim) != seen.end()) return true;
        seen.push_back(x.dim);
      }
    }
    return false;
  };
  for (std::size_t d = nd; d-- > 0;) {
    bool bad = nested(dec.output);
    for (const Placements& p : dec.inputs) bad = bad || nested(p);
    if (!bad) break;
    for (Placements& p : dec.inputs) p[d] = Placement::replicate();
    dec.output[d] = Placement::replicate();
    dec.rules[d] = "ic ice parcalama -> R";
  }
  return dec;
}

// ==========================================================================================
// dt_* islemleri
// ==========================================================================================

namespace {

DTensor make_out(std::vector<float> local, std::vector<std::int64_t> shape, DeviceMesh& mesh, Placements p) {
  return DTensor::from_local(std::move(local), std::move(shape), mesh, std::move(p));
}

}  // namespace

DTensor dt_matmul(const DTensor& a, const DTensor& b) {
  if (a.shape().size() != 2 || b.shape().size() != 2 || a.shape()[1] != b.shape()[0]) {
    throw DistError("Nexus dt_matmul: A [m,k], B [k,n] bekler");
  }
  const ShardingDecision dec = propagate(DOp::Matmul, {a.placements(), b.placements()});
  const DTensor a2 = a.redistribute(dec.inputs[0]);
  const DTensor b2 = b.redistribute(dec.inputs[1]);
  const std::vector<std::int64_t> as = a2.local_shape();
  const std::vector<std::int64_t> bs = b2.local_shape();
  if (as[1] != bs[0]) throw DistError("Nexus dt_matmul: yerel daralan eksenler uyusmuyor");
  std::vector<float> c(static_cast<std::size_t>(as[0] * bs[1]), 0.0f);
  for (std::int64_t i = 0; i < as[0]; ++i) {
    for (std::int64_t p = 0; p < as[1]; ++p) {
      const float av = a2.local()[static_cast<std::size_t>(i * as[1] + p)];
      const float* br = b2.local().data() + p * bs[1];
      float* cr = c.data() + i * bs[1];
      for (std::int64_t j = 0; j < bs[1]; ++j) cr[j] += av * br[j];
    }
  }
  return make_out(std::move(c), {a.shape()[0], b.shape()[1]}, a.mesh(), dec.output);
}

namespace {

DTensor elementwise(const DTensor& a, const DTensor& b, DOp op) {
  if (a.shape() != b.shape()) throw DistError("Nexus dt_add/dt_mul: ayni sekil bekler (yayin yok)");
  const ShardingDecision dec = propagate(op, {a.placements(), b.placements()});
  const DTensor a2 = a.redistribute(dec.inputs[0]);
  const DTensor b2 = b.redistribute(dec.inputs[1]);
  std::vector<float> c(a2.local().size());
  for (std::size_t i = 0; i < c.size(); ++i) {
    c[i] = op == DOp::Add ? a2.local()[i] + b2.local()[i] : a2.local()[i] * b2.local()[i];
  }
  return make_out(std::move(c), a.shape(), a.mesh(), dec.output);
}

}  // namespace

DTensor dt_add(const DTensor& a, const DTensor& b) { return elementwise(a, b, DOp::Add); }
DTensor dt_mul(const DTensor& a, const DTensor& b) { return elementwise(a, b, DOp::Mul); }

DTensor dt_layer_norm(const DTensor& x, const std::vector<float>& gamma, const std::vector<float>& beta, float eps) {
  if (x.shape().size() != 2) throw DistError("Nexus dt_layer_norm: [N, C] bekler");
  const ShardingDecision dec = propagate(DOp::Norm, {x.placements()});
  const DTensor x2 = x.redistribute(dec.inputs[0]);
  const std::vector<std::int64_t> ls = x2.local_shape();
  const std::int64_t c = ls[1];
  std::vector<float> y(x2.local().size());
  for (std::int64_t r = 0; r < ls[0]; ++r) {
    const float* xr = x2.local().data() + r * c;
    double mean = 0.0;
    for (std::int64_t i = 0; i < c; ++i) mean += xr[i];
    mean /= static_cast<double>(c);
    double var = 0.0;
    for (std::int64_t i = 0; i < c; ++i) var += (xr[i] - mean) * (xr[i] - mean);
    var /= static_cast<double>(c);
    const double inv = 1.0 / std::sqrt(var + eps);
    for (std::int64_t i = 0; i < c; ++i) {
      y[static_cast<std::size_t>(r * c + i)] =
          static_cast<float>((xr[i] - mean) * inv) * gamma[static_cast<std::size_t>(i)] + beta[static_cast<std::size_t>(i)];
    }
  }
  return make_out(std::move(y), x.shape(), x.mesh(), dec.output);
}

DTensor dt_softmax(const DTensor& x, int dim) {
  if (dim < 0) dim += static_cast<int>(x.shape().size());
  const ShardingDecision dec = propagate(DOp::Softmax, {x.placements()}, dim);
  const DTensor x2 = x.redistribute(dec.inputs[0]);
  const std::vector<std::int64_t> ls = x2.local_shape();
  std::int64_t pre = 1, post = 1;
  for (int i = 0; i < dim; ++i) pre *= ls[static_cast<std::size_t>(i)];
  for (std::size_t i = static_cast<std::size_t>(dim) + 1; i < ls.size(); ++i) post *= ls[i];
  const std::int64_t n = ls[static_cast<std::size_t>(dim)];
  std::vector<float> y(x2.local().size());
  for (std::int64_t p = 0; p < pre; ++p) {
    for (std::int64_t q = 0; q < post; ++q) {
      float m = -std::numeric_limits<float>::infinity();
      for (std::int64_t i = 0; i < n; ++i) m = std::max(m, x2.local()[static_cast<std::size_t>((p * n + i) * post + q)]);
      double s = 0.0;
      for (std::int64_t i = 0; i < n; ++i) s += std::exp(x2.local()[static_cast<std::size_t>((p * n + i) * post + q)] - m);
      for (std::int64_t i = 0; i < n; ++i) {
        const std::size_t at = static_cast<std::size_t>((p * n + i) * post + q);
        y[at] = static_cast<float>(std::exp(x2.local()[at] - m) / s);
      }
    }
  }
  return make_out(std::move(y), x.shape(), x.mesh(), dec.output);
}

DTensor dt_embedding(const DTensor& table, const std::vector<std::int32_t>& ids) {
  if (table.shape().size() != 2) throw DistError("Nexus dt_embedding: tablo [V, C] bekler");
  const Placements ids_p(table.placements().size(), Placement::replicate());
  const ShardingDecision dec = propagate(DOp::Embedding, {table.placements(), ids_p});
  const DTensor t2 = table.redistribute(dec.inputs[0]);
  const std::vector<std::int64_t> ls = t2.local_shape();
  const std::vector<std::int64_t> lo = DTensor::local_offset_of(t2.shape(), t2.mesh(), t2.placements());
  const std::int64_t c = ls[1];
  std::vector<float> y(ids.size() * static_cast<std::size_t>(c), 0.0f);
  for (std::size_t r = 0; r < ids.size(); ++r) {
    const std::int64_t v = ids[r] - lo[0];
    if (v < 0 || v >= ls[0]) continue;  // sozluk parcasi disinda: 0 (Partial)
    std::memcpy(y.data() + r * static_cast<std::size_t>(c), t2.local().data() + v * c, static_cast<std::size_t>(c) * sizeof(float));
  }
  return make_out(std::move(y), {static_cast<std::int64_t>(ids.size()), table.shape()[1]}, table.mesh(), dec.output);
}

}  // namespace dist
}  // namespace nexus
