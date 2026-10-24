/** FAZ 6 benchmark: SwiGLU MLP fwd+bwd (d=4096, d_ff=11008, BF16) ve RoPE cekirdegi GB/s. */
#include "nexus/cuda/gemm.hpp"
#include "nexus/cuda/kernels.hpp"
#include "nexus/cuda/lm_kernels.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <sstream>
#include <string>
#include <vector>
namespace nc = nexus::cuda;
using nexus::DType;
namespace {
struct Stats { double median = 0, min = 0, max = 0; int n = 0; };
Stats time_gpu(const nc::Stream& stream, const std::function<void()>& fn, int warmup = 10, int reps = 10) {
  for (int i = 0; i < warmup; ++i) fn();
  stream.synchronize();
  nc::Event a, b;
  std::vector<double> v;
  for (int i = 0; i < reps; ++i) {
    a.record(stream); fn(); b.record(stream); b.synchronize(); v.push_back(a.elapsed_ms(b));
  }
  std::sort(v.begin(), v.end());
  return {v[v.size() / 2], v.front(), v.back(), static_cast<int>(v.size())};
}
std::string json(const Stats& s) {
  std::ostringstream o; o.precision(7);
  o << "{\"median\":" << s.median << ",\"min\":" << s.min << ",\"max\":" << s.max << ",\"repeats\":" << s.n
    << ",\"unit\":\"ms\"}";
  return o.str();
}
void fill_bf16(void* dst, std::size_t n, float v, const nc::Stream& s) {
  nc::DeviceBuffer f(n * 4, s);
  nc::fill(f.as<float>(), v, n, s.handle());
  nc::cast(dst, DType::BF16, f.data(), DType::F32, n, s.handle());
}
nc::GemmArgs mm(bool ta, bool tb, int m, int n, int k, const void* a, int lda, const void* b, int ldb, void* c, int ldc,
                DType ct) {
  nc::GemmArgs g;
  g.trans_a = ta; g.trans_b = tb; g.m = m; g.n = n; g.k = k; g.a = a; g.lda = lda; g.b = b; g.ldb = ldb;
  g.c = c; g.ldc = ldc; g.a_type = DType::BF16; g.b_type = DType::BF16; g.c_type = ct;
  return g;
}

void swiglu_mlp(std::ostringstream& out, const nc::Stream& s) {
  constexpr int N = 4096, d = 4096, F = 11008;
  const std::size_t nx = static_cast<std::size_t>(N) * d, nup = static_cast<std::size_t>(d) * 2 * F,
                    nd = static_cast<std::size_t>(F) * d, nh = static_cast<std::size_t>(N) * 2 * F,
                    na = static_cast<std::size_t>(N) * F;
  nc::DeviceBuffer x(nx * 2, s), wup(nup * 2, s), wd(nd * 2, s), up(nh * 2, s), act(na * 2, s), y(nx * 2, s),
      dy(nx * 2, s), dact(na * 2, s), dup(nh * 2, s), dx(nx * 2, s), dwup(nup * 4, s), dwd(nd * 4, s);
  fill_bf16(x.data(), nx, 0.5f, s); fill_bf16(wup.data(), nup, 0.01f, s); fill_bf16(wd.data(), nd, 0.01f, s);
  fill_bf16(dy.data(), nx, 0.01f, s);
  cudaStream_t h = s.handle();
  auto gemm_fwd1 = [&] { nc::gemm(mm(false, false, N, 2 * F, d, x.data(), d, wup.data(), 2 * F, up.data(), 2 * F, DType::BF16), h); };
  auto gemm_fwd2 = [&] { nc::gemm(mm(false, false, N, d, F, act.data(), F, wd.data(), d, y.data(), d, DType::BF16), h); };
  auto fwd = [&] {
    gemm_fwd1();
    nc::gated_forward(act.data(), up.data(), DType::BF16, N, F, nc::GatedAct::SiLU, h);
    gemm_fwd2();
  };
  auto bwd_gemms_a = [&] {
    nc::gemm(mm(false, true, N, F, d, dy.data(), d, wd.data(), d, dact.data(), F, DType::BF16), h);   // dact = dy Wd^T
    nc::gemm(mm(true, false, F, d, N, act.data(), F, dy.data(), d, dwd.data(), d, DType::F32), h);    // dWd = act^T dy
  };
  auto bwd_gemms_b = [&] {
    nc::gemm(mm(false, true, N, d, 2 * F, dup.data(), 2 * F, wup.data(), 2 * F, dx.data(), d, DType::BF16), h);  // dx
    nc::gemm(mm(true, false, d, 2 * F, N, x.data(), d, dup.data(), 2 * F, dwup.data(), 2 * F, DType::F32), h);   // dWup
  };
  auto bwd = [&] {
    bwd_gemms_a();
    nc::gated_backward(dup.data(), dact.data(), up.data(), DType::BF16, N, F, nc::GatedAct::SiLU, h);
    bwd_gemms_b();
  };
  const Stats sf = time_gpu(s, fwd), sb = time_gpu(s, bwd), sfb = time_gpu(s, [&] { fwd(); bwd(); });
  const Stats sg = time_gpu(s, [&] { gemm_fwd1(); gemm_fwd2(); bwd_gemms_a(); bwd_gemms_b(); });
  const Stats sk = time_gpu(s, [&] {
    nc::gated_forward(act.data(), up.data(), DType::BF16, N, F, nc::GatedAct::SiLU, h);
    nc::gated_backward(dup.data(), dact.data(), up.data(), DType::BF16, N, F, nc::GatedAct::SiLU, h);
  });
  const double flop_f = 6.0 * N * static_cast<double>(d) * F;
  const double tf = flop_f / (sf.median * 1e9), tb = 2 * flop_f / (sb.median * 1e9), tfb = 3 * flop_f / (sfb.median * 1e9);
  // SwiGLU cekirdekleri bellek siniri: ileri 2F okur + F yazar, geri 2F+F okur + 2F yazar (BF16).
  const double kbytes = (3.0 + 5.0) * na * 2;
  std::printf("  info  SwiGLU MLP N=%d d=%d F=%d BF16: ileri %.2f ms (%.1f TF) geri %.2f ms (%.1f TF) ileri+geri %.2f ms (%.1f TF)\n",
              N, d, F, sf.median, tf, sb.median, tb, sfb.median, tfb);
  std::printf("  info    yalniz 6 GEMM %.2f ms (%.1f TF) | SwiGLU ileri+geri cekirdekleri %.3f ms (%.0f GB/s, toplamin %%%.1f'i)\n",
              sg.median, 3 * flop_f / (sg.median * 1e9), sk.median, kbytes / (sk.median * 1e6), 100.0 * sk.median / sfb.median);
  out << "  \"swiglu_mlp\":{\"N\":" << N << ",\"d\":" << d << ",\"F\":" << F << ",\"fwd\":" << json(sf) << ",\"bwd\":" << json(sb)
      << ",\"fwd_bwd\":" << json(sfb) << ",\"gemms_only\":" << json(sg) << ",\"gated_kernels\":" << json(sk)
      << ",\"fwd_tflops\":" << tf << ",\"bwd_tflops\":" << tb << ",\"fwd_bwd_tflops\":" << tfb
      << ",\"gated_kernels_gbps\":" << kbytes / (sk.median * 1e6) << "},\n";
}

void rope(std::ostringstream& out, const nc::Stream& s) {
  constexpr int N = 16384, H = 32, Hk = 8, D = 128;
  const int width = (H + 2 * Hk) * D;
  nc::DeviceRopeTable table(D, 500000.0f, {}, 8192, s);
  table.ensure(4096);
  out << "  \"rope\":[";
  bool first = true;
  for (const DType t : {DType::BF16, DType::F32}) {
    const std::size_t es = nexus::size_of(t);
    nc::DeviceBuffer qkv(static_cast<std::size_t>(N) * width * es, s);
    NEXUS_CUDA_CHECK(cudaMemsetAsync(qkv.data(), 0, qkv.bytes(), s.handle()));
    nc::RopeArgs a;
    a.x = qkv.data(); a.dtype = t; a.rows = N; a.row_stride = width; a.heads = H + Hk; a.head_dim = D; a.time = 4096;
    const Stats st = time_gpu(s, [&] { nc::rope_inplace(a, table, s.handle()); });
    const double bytes = 2.0 * N * (H + Hk) * D * es + 2.0 * 4096 * (D / 2) * 4;  // Q+K oku/yaz + cos/sin
    const double gbps = bytes / (st.median * 1e6);
    std::printf("  info  RoPE %s N=%d H=%d Hkv=%d D=%d (qkv yerinde): %.3f ms, %.0f GB/s\n", t == DType::BF16 ? "bf16" : "f32 ",
                N, H, Hk, D, st.median, gbps);
    out << (first ? "" : ",") << "{\"dtype\":\"" << (t == DType::BF16 ? "bf16" : "f32") << "\",\"time\":" << json(st)
        << ",\"gbps\":" << gbps << "}";
    first = false;
  }
  out << "],\n";
}
}  // namespace
int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  if (nc::device_count() == 0) { std::printf("  SKIP  CUDA aygiti yok\n"); return 77; }
  try {
    nc::Stream stream = nc::Stream::create(0); nc::StreamGuard guard(stream);
    cudaDeviceProp prop{}; NEXUS_CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::ostringstream out; out.precision(7);
    out << "{\n  \"phase\":6,\"device\":\"" << prop.name << "\",\"protocol\":{\"warmup\":10,\"repeats\":10,\"timing\":\"CUDA events\"},\n";
    swiglu_mlp(out, stream);
    rope(out, stream);
    out << "  \"notes\":\"PyTorch CUDA kurulu degil: eager/compile karsilastirmasi yok. RoPE ayri cekirdek; flash yukleme fuzyonu yok.\"\n}\n";
    const std::filesystem::path dir(NEXUS_FAZ6_DOCS_DIR); std::filesystem::create_directories(dir);
    std::ofstream(dir / "bench_cuda_phase6.json", std::ios::binary) << out.str();
    std::printf("  OK    FAZ 6 benchmark JSON yazildi\n");
    stream.destroy(); return 0;
  } catch (const std::exception& e) { std::printf("  FAIL  %s\n", e.what()); return 1; }
}
