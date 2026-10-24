/**
 * @file test_lm_kernels.cpp
 * @brief FAZ 6 GPU: RoPE (qkv adimli, yerinde, kismi, olcekli) ve SwiGLU/GeGLU ileri/geri,
 *        CPU nexus::lm referansina karsi (F32 ve BF16).
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/cuda/kernels.hpp"
#include "nexus/cuda/lm_kernels.hpp"
#include "nexus/lm/ops.hpp"
#include "nexus_test.hpp"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace nc = nexus::cuda;
namespace lm = nexus::lm;
namespace ag = nexus::autograd;
using nexus::DType;

namespace {

nc::Stream g_stream;

std::vector<float> randn(std::size_t n, std::uint64_t seed) {
  return ag::Variable::randn(ag::Shape{{static_cast<std::int64_t>(n)}}, seed, false).values();
}

// Dar tipte gidis-donus (referans girisi ayni yuvarlanmis degerleri gorsun).
std::vector<float> round_to(const std::vector<float>& v, DType t) {
  if (t == DType::F32) {
    return v;
  }
  nc::DeviceBuffer a(v.size() * 4, g_stream), b(v.size() * 2, g_stream);
  NEXUS_CUDA_CHECK(cudaMemcpyAsync(a.data(), v.data(), v.size() * 4, cudaMemcpyHostToDevice, g_stream.handle()));
  nc::cast(b.data(), t, a.data(), DType::F32, v.size(), g_stream.handle());
  nc::cast(a.data(), DType::F32, b.data(), t, v.size(), g_stream.handle());
  std::vector<float> out(v.size());
  NEXUS_CUDA_CHECK(cudaMemcpyAsync(out.data(), a.data(), v.size() * 4, cudaMemcpyDeviceToHost, g_stream.handle()));
  g_stream.synchronize();
  return out;
}

struct Dev {
  nc::DeviceBuffer buf;
  DType t;
  std::size_t n;
  Dev(const std::vector<float>& v, DType type) : buf(v.size() * nexus::size_of(type), g_stream), t(type), n(v.size()) {
    nc::DeviceBuffer f(v.size() * 4, g_stream);
    NEXUS_CUDA_CHECK(cudaMemcpyAsync(f.data(), v.data(), v.size() * 4, cudaMemcpyHostToDevice, g_stream.handle()));
    nc::cast(buf.data(), t, f.data(), DType::F32, n, g_stream.handle());
    g_stream.synchronize();
  }
  Dev(std::size_t count, DType type) : buf(count * nexus::size_of(type), g_stream), t(type), n(count) {}
  std::vector<float> host() const {
    nc::DeviceBuffer f(n * 4, g_stream);
    nc::cast(f.data(), DType::F32, buf.data(), t, n, g_stream.handle());
    std::vector<float> out(n);
    NEXUS_CUDA_CHECK(cudaMemcpyAsync(out.data(), f.data(), n * 4, cudaMemcpyDeviceToHost, g_stream.handle()));
    g_stream.synchronize();
    return out;
  }
};

double rel(const std::vector<float>& a, const std::vector<float>& b) {
  double num = 0.0;
  double den = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    num += (static_cast<double>(a[i]) - b[i]) * (static_cast<double>(a[i]) - b[i]);
    den += static_cast<double>(b[i]) * b[i];
  }
  return std::sqrt(num / std::max(den, 1e-30));
}

void report(const std::string& name, double err, double tol) {
  std::printf("  info  %-56s hata=%.3e esik=%.1e\n", name.c_str(), err, tol);
  nexus_test::expect(err <= tol, name);
}

void test_rope() {
  nexus_test::section("test_rope_cuda: qkv adimli yerinde RoPE vs CPU lm::rope_apply");
  const int B = 2, T = 37, H = 4, Hk = 2, D = 64;
  const int rows = B * T;
  const int width = (H + 2 * Hk) * D;
  for (const DType t : {DType::F32, DType::BF16}) {
    for (const int rot : {64, 32}) {
      lm::RopeScaling s;
      s.type = lm::RopeScaling::Type::Llama3;
      s.factor = 8.0f;
      s.original_max_position = 16;
      nc::DeviceRopeTable table(rot, 500000.0f, s, 4096, g_stream);
      table.ensure(T + 5);
      const std::vector<float> x = round_to(randn(static_cast<std::size_t>(rows) * width, 3), t);
      Dev d(x, t);
      nc::RopeArgs a;
      a.x = d.buf.data();
      a.dtype = t;
      a.rows = rows;
      a.row_stride = width;
      a.col0 = 0;
      a.heads = H + Hk;  // Q ve K bitisik: tek cagri
      a.head_dim = D;
      a.time = T;
      a.offset = 5;
      nc::rope_inplace(a, table, g_stream.handle());
      const std::vector<float> got = d.host();
      // CPU referans: her satirin Q|K bolgesi [rows, (H+Hk)*D], konum t + 5.
      lm::RopeCache cache(rot, 500000.0f, s, 4096);
      cache.ensure(T + 5);
      std::vector<float> ref = x;
      std::vector<float> qk(static_cast<std::size_t>(rows) * (H + Hk) * D);
      std::vector<int> pos(static_cast<std::size_t>(rows));
      for (int r = 0; r < rows; ++r) {
        pos[static_cast<std::size_t>(r)] = r % T + 5;
        std::copy(x.begin() + static_cast<std::ptrdiff_t>(r) * width,
                  x.begin() + static_cast<std::ptrdiff_t>(r) * width + (H + Hk) * D,
                  qk.begin() + static_cast<std::ptrdiff_t>(r) * (H + Hk) * D);
      }
      lm::rope_apply(qk.data(), rows, H + Hk, D, pos.data(), cache, false);
      for (int r = 0; r < rows; ++r) {
        std::copy(qk.begin() + static_cast<std::ptrdiff_t>(r) * (H + Hk) * D,
                  qk.begin() + static_cast<std::ptrdiff_t>(r + 1) * (H + Hk) * D,
                  ref.begin() + static_cast<std::ptrdiff_t>(r) * width);
      }
      const std::string tag = std::string(t == DType::F32 ? " f32" : " bf16") + " rot=" + std::to_string(rot);
      report("rope ileri (llama3 olcekli, offset 5)" + tag, rel(got, ref), t == DType::F32 ? 1e-6 : 4e-3);
      bool v_same = true;
      for (int r = 0; r < rows; ++r) {
        for (int c = (H + Hk) * D; c < width; ++c) {
          v_same = v_same && got[static_cast<std::size_t>(r) * width + c] == x[static_cast<std::size_t>(r) * width + c];
        }
      }
      nexus_test::expect(v_same, "V sutunlari degismez" + tag);
      a.transpose = true;
      nc::rope_inplace(a, table, g_stream.handle());
      report("rope transpoz o rope = I" + tag, rel(d.host(), x), t == DType::F32 ? 1e-6 : 8e-3);
    }
  }
}

void test_gated() {
  nexus_test::section("test_swiglu_cuda: SwiGLU/GeGLU ileri/geri vs CPU lm::swiglu (autograd)");
  const int rows = 129, f = 300;
  for (const DType t : {DType::F32, DType::BF16}) {
    for (const nc::GatedAct act : {nc::GatedAct::SiLU, nc::GatedAct::GeLU}) {
      const std::vector<float> x = round_to(randn(static_cast<std::size_t>(rows) * 2 * f, 7), t);
      const std::vector<float> g = round_to(randn(static_cast<std::size_t>(rows) * f, 8), t);
      const ag::Variable xv = ag::Variable::make(x, ag::Shape{{rows, 2 * f}}, true);
      const ag::Variable y = act == nc::GatedAct::SiLU ? lm::swiglu(xv) : lm::geglu(xv);
      y.backward(ag::Variable::make(g, ag::Shape{{rows, f}}, false));
      Dev din(x, t), dg(g, t), dout(static_cast<std::size_t>(rows) * f, t), dx(x.size(), t);
      nc::gated_forward(dout.buf.data(), din.buf.data(), t, rows, f, act, g_stream.handle());
      nc::gated_backward(dx.buf.data(), dg.buf.data(), din.buf.data(), t, rows, f, act, g_stream.handle());
      const std::string tag = std::string(act == nc::GatedAct::SiLU ? " swiglu" : " geglu") +
                              (t == DType::F32 ? " f32" : " bf16");
      const double tol = t == DType::F32 ? 2e-6 : 8e-3;
      report("ileri" + tag, rel(dout.host(), y.values()), tol);
      report("geri" + tag, rel(dx.host(), xv.grad().values()), tol);
    }
  }
}

}  // namespace

int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  if (nc::device_count() == 0) {
    std::printf("  SKIP  CUDA aygiti yok\n");
    return 77;
  }
  try {
    g_stream = nc::Stream::create(0);
    nc::StreamGuard guard(g_stream);
    test_rope();
    test_gated();
    g_stream.synchronize();
    g_stream.destroy();
  } catch (const std::exception& e) {
    std::printf("  FAIL  %s\n", e.what());
    return 1;
  }
  return nexus_test::finish("faz6-lm-cuda");
}
