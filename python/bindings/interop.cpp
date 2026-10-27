/**
 * @file interop.cpp
 * @brief FAZ 17.5 / 17.6: Python bagi — nexus.interop (DLPack sifir kopya, HF / GGUF / ONNX / safetensors),
 *        nexus.lm (LanguageModel: logitler, KV onbellekli greedy uretim), nexus.config (on tanimlilar / JSON).
 *
 * DLPack protokolu: from_dlpack(obj) obj.__dlpack__() kapsulunu tuketir ("dltensor" -> "used_dltensor");
 * Tensor.__dlpack__() "dltensor" kapsulu verir, tuketilmezse kapsul yikicisi deleter'i cagirir. Uyumsuz dtype /
 * adim / cihaz acik hata (kopyaya dusme yok). CUDA tensorlerinde islemler nexus::cuda cekirdekleriyle yapilir
 * (modul CUDA'siz derlendiyse acik hata).
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/autograd/autograd.hpp"
#include "nexus/interop/dlpack.hpp"
#include "nexus/interop/gguf.hpp"
#include "nexus/interop/hf.hpp"
#include "nexus/interop/onnx.hpp"
#include "nexus/interop/safetensors.hpp"
#include "nexus/lm/config.hpp"
#include "nexus/lm/model.hpp"

#if defined(NEXUS_PY_CUDA)
#include "nexus/cuda/kernels.hpp"
#include <cuda_runtime.h>
#endif

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>

namespace py = pybind11;
namespace ix = nexus::interop;
namespace lm = nexus::lm;

namespace {

using TPtr = std::shared_ptr<nexus::Tensor>;

bool on_cuda(const nexus::Tensor& t) { return t.space() == nexus::MemorySpace::Device; }

void need_cuda(const char* what) {
#if !defined(NEXUS_PY_CUDA)
  throw std::runtime_error(std::string("nexus.interop: ") + what + ": modul CUDA'siz derlendi (CUDA tensoru islenemez)");
#else
  (void)what;
#endif
}

TPtr from_dlpack_obj(py::object obj) {
  py::object cap = py::hasattr(obj, "__dlpack__") ? obj.attr("__dlpack__")() : obj;
  PyObject* c = cap.ptr();
  if (!PyCapsule_IsValid(c, "dltensor")) throw std::invalid_argument("nexus.interop.from_dlpack: 'dltensor' kapsulu degil (zaten tuketilmis olabilir)");
  auto* m = static_cast<DLManagedTensor*>(PyCapsule_GetPointer(c, "dltensor"));
  if (PyCapsule_SetName(c, "used_dltensor") != 0) throw py::error_already_set();
  return std::make_shared<nexus::Tensor>(ix::from_dlpack(m));
}

py::capsule to_capsule(const TPtr& t) {
  DLManagedTensor* m = ix::to_dlpack(t);
  PyObject* cap = PyCapsule_New(m, "dltensor", [](PyObject* self) {
    if (PyCapsule_IsValid(self, "dltensor")) {  // tuketilmedi: sahiplik bizde kaldi
      auto* mt = static_cast<DLManagedTensor*>(PyCapsule_GetPointer(self, "dltensor"));
      if (mt && mt->deleter) mt->deleter(mt);
    }
  });
  if (cap == nullptr) {
    m->deleter(m);
    throw py::error_already_set();
  }
  return py::reinterpret_steal<py::capsule>(cap);
}

std::string device_str(const nexus::Tensor& t) {
  if (on_cuda(t)) return "cuda:" + std::to_string(t.device().index < 0 ? 0 : t.device().index);
  return t.space() == nexus::MemorySpace::PinnedHost ? "cpu(pinned)" : "cpu";
}

std::vector<std::int32_t> as_tokens(py::object tokens) {
  py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> a(tokens);
  std::vector<std::int32_t> t(static_cast<std::size_t>(a.size()));
  const std::int64_t* p = a.data();
  for (std::size_t i = 0; i < t.size(); ++i) t[i] = static_cast<std::int32_t>(p[i]);
  return t;
}

ix::GgufType gguf_type(const std::string& s) {
  if (s == "F32") return ix::GgufType::F32;
  if (s == "F16") return ix::GgufType::F16;
  if (s == "Q8_0") return ix::GgufType::Q8_0;
  if (s == "Q4_0") return ix::GgufType::Q4_0;
  throw std::invalid_argument("GGUF tipi F32 | F16 | Q8_0 | Q4_0");
}

}  // namespace

void bind_faz17(py::module_& m) {
  py::module_ io = m.def_submodule("interop", "FAZ 17: DLPack, safetensors, HF, GGUF, ONNX");
  py::class_<nexus::Tensor, TPtr>(io, "Tensor", "float32 nexus::Tensor (CPU ya da CUDA); DLPack ile sifir kopya")
      .def_property_readonly("shape", [](const nexus::Tensor& t) {
        std::vector<std::size_t> s;
        for (std::uint8_t i = 0; i < t.shape().rank(); ++i) s.push_back(t.shape()[i]);
        return s;
      })
      .def_property_readonly("device", &device_str)
      .def_property_readonly("data_ptr", [](const nexus::Tensor& t) { return reinterpret_cast<std::uintptr_t>(t.data()); })
      .def("numel", &nexus::Tensor::numel)
      .def("__dlpack__", [](const TPtr& self, py::object stream) {
            (void)stream;  // nexus islemleri donmeden once senkron: veri her akista hazir
            return to_capsule(self);
          }, py::arg("stream") = py::none())
      .def("__dlpack_device__", [](const nexus::Tensor& t) {
        if (on_cuda(t)) return py::make_tuple(static_cast<int>(kDLCUDA), t.device().index < 0 ? 0 : t.device().index);
        return py::make_tuple(static_cast<int>(t.space() == nexus::MemorySpace::PinnedHost ? kDLCUDAHost : kDLCPU), 0);
      })
      .def("numpy", [](const TPtr& self) {
            if (on_cuda(*self)) throw std::runtime_error("nexus.interop.Tensor.numpy: CUDA tensoru (once torch.from_dlpack kullanin)");
            std::vector<py::ssize_t> shape, strides;
            py::ssize_t st = 4;
            for (int i = self->shape().rank() - 1; i >= 0; --i) {
              shape.insert(shape.begin(), static_cast<py::ssize_t>(self->shape()[static_cast<std::size_t>(i)]));
              strides.insert(strides.begin(), st);
              st *= static_cast<py::ssize_t>(self->shape()[static_cast<std::size_t>(i)]);
            }
            auto* keep = new TPtr(self);
            py::capsule base(keep, [](void* p) { delete static_cast<TPtr*>(p); });
            return py::array_t<float>(shape, strides, self->data(), base);  // sifir kopya goruntu
          })
      .def("add_", [](nexus::Tensor& self, const nexus::Tensor& other) -> nexus::Tensor& {
            if (self.numel() != other.numel() || on_cuda(self) != on_cuda(other)) throw std::invalid_argument("add_: ayni boy ve cihaz");
            if (on_cuda(self)) {
              need_cuda("add_");
#if defined(NEXUS_PY_CUDA)
              nexus::cuda::add_inplace(self.data(), other.data(), self.numel(), nullptr);
              if (cudaDeviceSynchronize() != cudaSuccess) throw std::runtime_error("add_: CUDA hatasi");
#endif
            } else {
              float* a = self.data();
              const float* b = other.data();
              for (std::size_t i = 0; i < self.numel(); ++i) a[i] += b[i];
            }
            return self;
          }, py::return_value_policy::reference_internal)
      .def("scale_", [](nexus::Tensor& self, float s) -> nexus::Tensor& {
            if (on_cuda(self)) throw std::runtime_error("scale_: CUDA'da yok (add_ kullanin)");
            float* a = self.data();
            for (std::size_t i = 0; i < self.numel(); ++i) a[i] *= s;
            return self;
          }, py::return_value_policy::reference_internal)
      .def("sum", [](const nexus::Tensor& self) {
        if (on_cuda(self)) {
          need_cuda("sum");
#if defined(NEXUS_PY_CUDA)
          std::vector<float> h(self.numel());
          if (cudaMemcpy(h.data(), self.data(), h.size() * 4, cudaMemcpyDeviceToHost) != cudaSuccess) throw std::runtime_error("sum: CUDA kopya");
          double s = 0;
          for (float v : h) s += v;
          return s;
#endif
        }
        double s = 0;
        for (std::size_t i = 0; i < self.numel(); ++i) s += self.data()[i];
        return s;
      });
  io.def("from_dlpack", &from_dlpack_obj, py::arg("obj"), "DLPack ureticisinden (orn. torch tensoru) sifir kopya nexus Tensor");
  io.def("load_safetensors", [](const std::string& path, bool mmap) {
    const ix::SafeTensors st = ix::SafeTensors::open(path, mmap);
    py::dict d;
    for (const ix::StTensor& t : st.tensors()) {
      std::vector<py::ssize_t> shape(t.shape.begin(), t.shape.end());
      if (t.dtype == "I64") {
        py::array_t<std::int64_t> a(shape);
        std::memcpy(a.mutable_data(), st.bytes(t).data(), static_cast<std::size_t>(t.end - t.begin));
        d[py::str(t.name)] = a;
      } else {
        const auto v = st.to_f32(t);
        py::array_t<float> a(shape);
        std::memcpy(a.mutable_data(), v.data(), v.size() * 4);
        d[py::str(t.name)] = a;
      }
    }
    return d;
  }, py::arg("path"), py::arg("mmap") = true);
  io.def("save_safetensors", [](const std::string& path, py::dict tensors) {
    std::vector<py::array_t<float, py::array::c_style | py::array::forcecast>> keep;
    std::vector<ix::StWrite> w;
    for (auto item : tensors) {
      keep.emplace_back(py::reinterpret_borrow<py::object>(item.second));
      const auto& a = keep.back();
      std::vector<std::int64_t> shape(a.shape(), a.shape() + a.ndim());
      w.push_back(ix::StWrite{py::cast<std::string>(item.first), "F32", shape, a.data(), static_cast<std::size_t>(a.size()) * 4});
    }
    ix::write_safetensors(path, w, {{"producer", "nexus-python"}});
  });

  py::module_ lmm = m.def_submodule("lm", "FAZ 6 / 16 / 17: dil modeli");
  py::class_<lm::LanguageModel, std::shared_ptr<lm::LanguageModel>>(lmm, "LanguageModel")
      .def("num_parameters", &lm::LanguageModel::num_parameters)
      .def("config_json", [](const lm::LanguageModel& self) { return lm::to_json(self.config()); })
      .def("logits", [](const lm::LanguageModel& self, py::object tokens, int batch) {
            const auto t = as_tokens(tokens);
            if (batch < 1 || t.size() % static_cast<std::size_t>(batch) != 0) throw std::invalid_argument("logits: token sayisi batch'e bolunmeli");
            nexus::autograd::NoGradGuard ng;
            const auto v = self.logits(t, batch, static_cast<int>(t.size()) / batch).values();
            const std::size_t Vp = v.size() / t.size(), V = static_cast<std::size_t>(self.config().vocab_size);
            py::array_t<float> a({static_cast<py::ssize_t>(t.size()), static_cast<py::ssize_t>(V)});
            for (std::size_t r = 0; r < t.size(); ++r) std::memcpy(a.mutable_data() + r * V, v.data() + r * Vp, V * 4);
            return a;
          }, py::arg("tokens"), py::arg("batch") = 1)
      .def("generate", [](const lm::LanguageModel& self, py::object prompt, int max_new) {
            nexus::autograd::NoGradGuard ng;
            lm::KvCache cache = self.make_cache();
            std::vector<std::int32_t> feed = as_tokens(prompt), out;
            lm::ForwardOptions o;
            o.cache = &cache;
            for (int i = 0; i < max_new; ++i) {
              const auto v = self.logits(feed, 1, static_cast<int>(feed.size()), o).values();
              const std::size_t Vp = v.size() / feed.size();
              const float* last = v.data() + (feed.size() - 1) * Vp;
              const std::int32_t nx = static_cast<std::int32_t>(std::max_element(last, last + self.config().vocab_size) - last);
              out.push_back(nx);
              feed = {nx};
            }
            return out;
          }, py::arg("prompt"), py::arg("max_new"), "greedy, KV onbellekli (FAZ 16)");
  lmm.def("build", [](const std::string& preset, const std::string& overrides) { return std::make_shared<lm::LanguageModel>(lm::build(preset, overrides)); },
          py::arg("preset"), py::arg("overrides_json") = "{}");
  lmm.def("from_json", [](const std::string& json) { return std::make_shared<lm::LanguageModel>(lm::from_json(json)); });

  py::module_ cfg = m.def_submodule("config", "lm yapilandirmalari");
  cfg.def("preset", [](const std::string& name) { return lm::to_json(lm::preset(name)); });
  cfg.def("count_parameters", [](const std::string& json) { return lm::count_parameters(lm::from_json(json)); });

  io.def("load_hf", [](const std::string& dir, bool mmap) {
    ix::HfLoadStats st;
    auto mdl = std::make_shared<lm::LanguageModel>(ix::load_hf(dir, mmap, &st));
    return py::make_tuple(mdl, py::dict(py::arg("template") = st.template_name, py::arg("tensors") = st.tensors,
                                         py::arg("bytes") = st.bytes, py::arg("ms") = st.ms));
  }, py::arg("dir"), py::arg("mmap") = true);
  io.def("save_hf", [](const lm::LanguageModel& mdl, const std::string& dir, const std::string& dtype) { ix::save_hf(mdl, dir, dtype); },
         py::arg("model"), py::arg("dir"), py::arg("dtype") = "F32");
  io.def("export_gguf", [](const lm::LanguageModel& mdl, const std::string& path, const std::string& type, const std::string& tok) {
    ix::GgufOptions o;
    o.type = gguf_type(type);
    o.tokenizer_json = tok;
    ix::GgufStats st;
    ix::export_gguf(mdl, path, o, &st);
    return py::dict(py::arg("bytes") = st.bytes, py::arg("tensors") = st.tensors, py::arg("ms") = st.ms);
  }, py::arg("model"), py::arg("path"), py::arg("type") = "F16", py::arg("tokenizer_json") = "");
  io.def("import_gguf", [](const std::string& path) { return std::make_shared<lm::LanguageModel>(ix::import_gguf(path)); });
  io.def("export_onnx", [](const lm::LanguageModel& mdl, const std::string& path, int opset, bool external) {
    ix::OnnxOptions o;
    o.opset = opset;
    o.external_data = external;
    ix::OnnxStats st;
    ix::export_onnx(mdl, path, o, &st);
    return py::dict(py::arg("nodes") = st.nodes, py::arg("model_bytes") = st.model_bytes, py::arg("data_bytes") = st.data_bytes);
  }, py::arg("model"), py::arg("path"), py::arg("opset") = 17, py::arg("external_data") = false);
}
