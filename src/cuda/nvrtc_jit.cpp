/**
 * @file nvrtc_jit.cpp
 * @brief FAZ 13.4: NVRTC ile CUDA fuzyon cekirdegi derleme, cubin disk onbellegi, surucu API'siyle yukleme/baslatma.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/cuda/nvrtc_jit.hpp"

#include <cuda.h>
#include <nvrtc.h>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>

namespace nexus {
namespace cuda {

namespace fs = std::filesystem;

namespace {

void nv(nvrtcResult r, const char* what) {
  if (r != NVRTC_SUCCESS) throw std::runtime_error(std::string("Nexus NVRTC: ") + what + ": " + nvrtcGetErrorString(r));
}

void cu(CUresult r, const char* what) {
  if (r != CUDA_SUCCESS) {
    const char* msg = nullptr;
    cuGetErrorString(r, &msg);
    throw std::runtime_error(std::string("Nexus CUDA surucu: ") + what + ": " + (msg ? msg : "?"));
  }
}

void rt(cudaError_t e, const char* what) {
  if (e != cudaSuccess) throw std::runtime_error(std::string("Nexus CUDA: ") + what + ": " + cudaGetErrorString(e));
}

}  // namespace

NvrtcJit::NvrtcJit(std::string cache_dir) : dir_(cache_dir.empty() ? ir::default_cache_dir() : std::move(cache_dir)) {
  fs::create_directories(dir_);
  rt(cudaFree(nullptr), "baglam");  // birincil baglami etkinlestir (surucu API'si icin)
}

NvrtcJit::~NvrtcJit() {
  for (void* m : modules_) cuModuleUnload(static_cast<CUmodule>(m));
}

void NvrtcJit::load(const std::string& source, const std::vector<std::string>& names) {
  int dev = 0;
  rt(cudaGetDevice(&dev), "cudaGetDevice");
  int major = 0, minor = 0;
  rt(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev), "cc");
  rt(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev), "cc");
  const std::string arch = "sm_" + std::to_string(major) + std::to_string(minor);
  const std::string h = ir::source_hash(source + "|nvrtc|" + arch + "|v1");
  const fs::path cubin_path = fs::path(dir_) / (h + ".cubin");
  std::vector<char> cubin;
  if (fs::exists(cubin_path)) {
    std::ifstream f(cubin_path, std::ios::binary);
    cubin.assign(std::istreambuf_iterator<char>(f), {});
    stats_.cache_hits += 1;
  } else {
    const auto t0 = std::chrono::steady_clock::now();
    nvrtcProgram prog = nullptr;
    nv(nvrtcCreateProgram(&prog, source.c_str(), (h + ".cu").c_str(), 0, nullptr, nullptr), "nvrtcCreateProgram");
    const std::string opt_arch = "--gpu-architecture=" + arch;
    const char* opts[] = {opt_arch.c_str(), "--std=c++17", "-default-device"};
    const nvrtcResult rc = nvrtcCompileProgram(prog, 3, opts);
    if (rc != NVRTC_SUCCESS) {
      std::size_t n = 0;
      nvrtcGetProgramLogSize(prog, &n);
      std::string log(n, '\0');
      nvrtcGetProgramLog(prog, log.data());
      nvrtcDestroyProgram(&prog);
      throw std::runtime_error("Nexus NVRTC: derleme basarisiz:\n" + log);
    }
    std::size_t n = 0;
    nv(nvrtcGetCUBINSize(prog, &n), "nvrtcGetCUBINSize");
    cubin.resize(n);
    nv(nvrtcGetCUBIN(prog, cubin.data()), "nvrtcGetCUBIN");
    nvrtcDestroyProgram(&prog);
    stats_.compile_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    stats_.compiler_invocations += 1;
    std::ofstream(cubin_path, std::ios::binary).write(cubin.data(), static_cast<std::streamsize>(cubin.size()));
  }
  CUmodule mod = nullptr;
  cu(cuModuleLoadData(&mod, cubin.data()), "cuModuleLoadData");
  modules_.push_back(mod);
  for (const std::string& name : names) {
    CUfunction fn = nullptr;
    cu(cuModuleGetFunction(&fn, mod, name.c_str()), ("cuModuleGetFunction " + name).c_str());
    funcs_[name] = fn;
  }
}

void NvrtcJit::launch(const std::string& name, const float* const* d_in, float* const* d_out, long long rows, long long cols,
                      cudaStream_t stream) {
  auto it = funcs_.find(name);
  if (it == funcs_.end()) throw std::runtime_error("Nexus NVRTC: yuklenmemis cekirdek " + name);
  const long long n = rows * cols;
  const unsigned threads = 256;
  const long long work = (cols & 3) == 0 ? n / 4 : n;
  const unsigned blocks = static_cast<unsigned>(std::clamp<long long>((work + threads - 1) / threads, 1, 65535LL * 8));
  void* args[] = {static_cast<void*>(&d_in), static_cast<void*>(&d_out), &rows, &cols};
  cu(cuLaunchKernel(static_cast<CUfunction>(it->second), blocks, 1, 1, threads, 1, 1, 0, reinterpret_cast<CUstream>(stream),
                    args, nullptr),
     "cuLaunchKernel");
}

void NvrtcJit::run(const std::string& name, const std::vector<std::vector<float>>& in, std::vector<std::vector<float>>& out,
                   long long rows, long long cols) {
  std::vector<float*> din(in.size()), dout(out.size());
  for (std::size_t i = 0; i < in.size(); ++i) {
    rt(cudaMalloc(reinterpret_cast<void**>(&din[i]), in[i].size() * 4), "cudaMalloc");
    rt(cudaMemcpy(din[i], in[i].data(), in[i].size() * 4, cudaMemcpyHostToDevice), "H2D");
  }
  for (std::size_t i = 0; i < out.size(); ++i) rt(cudaMalloc(reinterpret_cast<void**>(&dout[i]), out[i].size() * 4), "cudaMalloc");
  float** pin = nullptr;
  float** pout = nullptr;
  rt(cudaMalloc(reinterpret_cast<void**>(&pin), std::max<std::size_t>(1, din.size()) * sizeof(float*)), "cudaMalloc");
  rt(cudaMalloc(reinterpret_cast<void**>(&pout), std::max<std::size_t>(1, dout.size()) * sizeof(float*)), "cudaMalloc");
  if (!din.empty()) rt(cudaMemcpy(pin, din.data(), din.size() * sizeof(float*), cudaMemcpyHostToDevice), "H2D");
  rt(cudaMemcpy(pout, dout.data(), dout.size() * sizeof(float*), cudaMemcpyHostToDevice), "H2D");
  launch(name, pin, pout, rows, cols, nullptr);
  rt(cudaDeviceSynchronize(), "senkron");
  for (std::size_t i = 0; i < out.size(); ++i) {
    rt(cudaMemcpy(out[i].data(), dout[i], out[i].size() * 4, cudaMemcpyDeviceToHost), "D2H");
  }
  for (float* p : din) cudaFree(p);
  for (float* p : dout) cudaFree(p);
  cudaFree(pin);
  cudaFree(pout);
}

}  // namespace cuda
}  // namespace nexus
