/**
 * @file test_parity_nano.cpp
 * @brief FAZ 0.E parity duzenegi: Nexus (autograd + fuzyonlu op'lar) ile PyTorch
 *        ayni agirlik, ayni batch, ayni tohumla; kayip ve TUM parametre gradyanlari.
 *
 * Akis:
 *  1. configs/<ad>.json okunur (C++ ve Python ayni dosyayi okur).
 *  2. Nexus tarafinda kucuk GPT kurulur: embedding_pos -> L x [cok basli nedensel
 *     dikkat -> residual + LayerNorm -> GELU MLP -> residual + LayerNorm] -> Linear
 *     -> kaydirilmis CE. (Nexus CausalTransformerBlock'un post-LN tarifi.)
 *  3. Agirliklar, tokenlar, Nexus kaybi ve gradyanlari gecici klasore yazilir.
 *  4. reference_torch.py ayni modeli PyTorch FP32'de kurar ve gradyanlari yazar;
 *     compare.py max_abs, max_rel, kosinus ve ULP histogramini hesaplar.
 *  5. Esikler (promt.txt): kayip max_rel < 1e-5, gradyan max_rel < 1e-4.
 *
 * PyTorch (veya Python) yoksa test 77 ile cikar ve ctest onu SKIPPED sayar,
 * "passed" degil.
 *
 * Kullanim: test_parity_nano <config.json> [batch] [time]
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "json_mini.hpp"
#include "nexus/autograd/autograd.hpp"
#include "nexus/fused/fused.hpp"
#include "nexus/fused/training.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#if !defined(_WIN32)
#include <sys/wait.h>
#endif

namespace ag = nexus::autograd;
namespace fu = nexus::fused;
using ag::Shape;
using ag::Variable;

namespace {

constexpr int kSkip = 77;

struct Config {
  std::string name;
  int vocab = 0;
  int dim = 0;
  int heads = 0;
  int ff = 0;
  int layers = 0;
  int batch = 0;
  int time = 0;
  std::uint64_t seed = 0;
};

struct Param {
  std::string name;
  Variable value;
};

// Cikis kodunu dondurur. Windows'ta cmd /c dis tirnaklari soyar; komutun tamami
// bir kat daha tirnaklanir.
[[nodiscard]] int run_status(const std::string& command) {
#if defined(_WIN32)
  return std::system(("\"" + command + "\"").c_str());
#else
  const int code = std::system(command.c_str());
  return WIFEXITED(code) ? WEXITSTATUS(code) : -1;
#endif
}

[[nodiscard]] bool run_command(const std::string& command) { return run_status(command) == 0; }

class Model {
 public:
  explicit Model(const Config& c) : c_(c) {
    std::mt19937_64 rng(c.seed);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    auto make = [&](const std::string& name, std::int64_t rows, std::int64_t cols, float scale, float shift) {
      Shape shape = cols > 0 ? Shape{{rows, cols}} : Shape{{rows}};
      std::vector<float> values(static_cast<std::size_t>(shape.numel()));
      for (float& v : values) {
        v = shift + scale * normal(rng);
      }
      params_.push_back({name, Variable::make(std::move(values), std::move(shape), true)});
      return params_.back().value;
    };
    const float w = 0.02f;
    wte_ = make("wte", c.vocab, c.dim, w, 0.0f);
    wpe_ = make("wpe", c.time, c.dim, w, 0.0f);
    for (int l = 0; l < c.layers; ++l) {
      const std::string p = "h" + std::to_string(l) + ".";
      Layer layer;
      layer.wq = make(p + "wq", c.dim, c.dim, w, 0.0f);
      layer.bq = make(p + "bq", c.dim, 0, w, 0.0f);
      layer.wk = make(p + "wk", c.dim, c.dim, w, 0.0f);
      layer.bk = make(p + "bk", c.dim, 0, w, 0.0f);
      layer.wv = make(p + "wv", c.dim, c.dim, w, 0.0f);
      layer.bv = make(p + "bv", c.dim, 0, w, 0.0f);
      layer.wo = make(p + "wo", c.dim, c.dim, w, 0.0f);
      layer.bo = make(p + "bo", c.dim, 0, w, 0.0f);
      layer.ln1_g = make(p + "ln1_g", c.dim, 0, 0.1f, 1.0f);
      layer.ln1_b = make(p + "ln1_b", c.dim, 0, 0.1f, 0.0f);
      layer.w1 = make(p + "w1", c.dim, c.ff, w, 0.0f);
      layer.b1 = make(p + "b1", c.ff, 0, w, 0.0f);
      layer.w2 = make(p + "w2", c.ff, c.dim, w, 0.0f);
      layer.b2 = make(p + "b2", c.dim, 0, w, 0.0f);
      layer.ln2_g = make(p + "ln2_g", c.dim, 0, 0.1f, 1.0f);
      layer.ln2_b = make(p + "ln2_b", c.dim, 0, 0.1f, 0.0f);
      layers_.push_back(layer);
    }
    head_w_ = make("head_w", c.dim, c.vocab, w, 0.0f);
    head_b_ = make("head_b", c.vocab, 0, w, 0.0f);
    std::vector<float> mask(static_cast<std::size_t>(c.time) * static_cast<std::size_t>(c.time), 0.0f);
    for (int i = 0; i < c.time; ++i) {
      for (int j = i + 1; j < c.time; ++j) {
        mask[static_cast<std::size_t>(i * c.time + j)] = -1.0e9f;  // reference_torch.py ile ayni sabit
      }
    }
    mask_ = Variable::make(std::move(mask), Shape{{c.time, c.time}}, false);
  }

  [[nodiscard]] Variable loss(const std::vector<std::int32_t>& tokens) const {
    const int T = c_.time;
    const int dh = c_.dim / c_.heads;
    const float scale = 1.0f / std::sqrt(static_cast<float>(dh));
    Variable x = fu::embedding_pos(wte_, wpe_, tokens, c_.batch, T);
    for (const Layer& layer : layers_) {
      const Variable q = ag::add_bias(ag::matmul(x, layer.wq), layer.bq);
      const Variable k = ag::add_bias(ag::matmul(x, layer.wk), layer.bk);
      const Variable v = ag::add_bias(ag::matmul(x, layer.wv), layer.bv);
      std::vector<Variable> rows;
      for (int b = 0; b < c_.batch; ++b) {
        const Variable qb = ag::narrow(q, 0, static_cast<std::int64_t>(b) * T, T);
        const Variable kb = ag::narrow(k, 0, static_cast<std::int64_t>(b) * T, T);
        const Variable vb = ag::narrow(v, 0, static_cast<std::int64_t>(b) * T, T);
        std::vector<Variable> heads;
        for (int h = 0; h < c_.heads; ++h) {
          const Variable qh = ag::narrow(qb, 1, static_cast<std::int64_t>(h) * dh, dh);
          const Variable kh = ag::narrow(kb, 1, static_cast<std::int64_t>(h) * dh, dh);
          const Variable vh = ag::narrow(vb, 1, static_cast<std::int64_t>(h) * dh, dh);
          const Variable scores = ag::add(ag::mul_scalar(ag::matmul(qh, ag::transpose(kh)), scale), mask_);
          heads.push_back(ag::matmul(fu::fused_softmax(scores), vh));
        }
        rows.push_back(ag::cat(heads, 1));
      }
      const Variable attn = ag::add_bias(ag::matmul(ag::cat(rows, 0), layer.wo), layer.bo);
      x = fu::fused_residual_layernorm(attn, x, layer.ln1_g, layer.ln1_b, 1.0e-5f);
      const Variable hidden = ag::gelu(ag::add_bias(ag::matmul(x, layer.w1), layer.b1));
      const Variable mlp = ag::add_bias(ag::matmul(hidden, layer.w2), layer.b2);
      x = fu::fused_residual_layernorm(mlp, x, layer.ln2_g, layer.ln2_b, 1.0e-5f);
    }
    const Variable logits = ag::add_bias(ag::matmul(x, head_w_), head_b_);
    return fu::cross_entropy_shifted(logits, tokens, c_.batch, T);
  }

  [[nodiscard]] const std::vector<Param>& params() const noexcept { return params_; }

 private:
  struct Layer {
    Variable wq, bq, wk, bk, wv, bv, wo, bo, ln1_g, ln1_b, w1, b1, w2, b2, ln2_g, ln2_b;
  };
  Config c_;
  std::vector<Param> params_;
  Variable wte_, wpe_, head_w_, head_b_, mask_;
  std::vector<Layer> layers_;
};

void write_floats(std::ofstream& out, const std::vector<float>& values) {
  out.write(reinterpret_cast<const char*>(values.data()), static_cast<std::streamsize>(values.size() * sizeof(float)));
}

std::string json_escape(const std::string& path) {
  std::string out;
  for (const char ch : path) {
    if (ch == '\\') {
      out += "\\\\";
    } else if (ch == '"') {
      out += "\\\"";
    } else {
      out += ch;
    }
  }
  return out;
}

int run(int argc, char** argv) {
  if (argc < 2) {
    std::printf("kullanim: test_parity_nano <config.json> [batch] [time]\n");
    return 2;
  }
  const std::string config_path = argv[1];
  const auto root = nexus::jsonmini::parse_file(config_path);
  const auto& model = root.at("model");
  const auto& data = root.at("data");
  Config c;
  c.name = nexus::jsonmini::get_string(root, "name", "config");
  c.vocab = static_cast<int>(nexus::jsonmini::require_int(model, "vocab"));
  c.dim = static_cast<int>(nexus::jsonmini::require_int(model, "dim"));
  c.heads = static_cast<int>(nexus::jsonmini::require_int(model, "heads"));
  c.ff = static_cast<int>(nexus::jsonmini::require_int(model, "ff"));
  c.layers = static_cast<int>(nexus::jsonmini::require_int(model, "layers"));
  c.batch = static_cast<int>(nexus::jsonmini::require_int(data, "batch"));
  c.time = static_cast<int>(nexus::jsonmini::require_int(data, "time"));
  c.seed = static_cast<std::uint64_t>(nexus::jsonmini::get_int(data, "seed", 1));
  if (argc >= 3) {
    c.batch = std::atoi(argv[2]);
  }
  if (argc >= 4) {
    c.time = std::atoi(argv[3]);
  }
  if (c.dim % c.heads != 0 || c.batch < 1 || c.time < 2) {
    std::printf("  FAIL  gecersiz yapilandirma\n");
    return 1;
  }
  std::printf("parity %s: L=%d d=%d h=%d ff=%d V=%d B=%d T=%d\n", c.name.c_str(), c.layers, c.dim, c.heads, c.ff,
              c.vocab, c.batch, c.time);

  // Python + PyTorch var mi? Yoksa SKIPPED (77), "passed" degil.
  const std::string python = NEXUS_PARITY_PYTHON;
  if (python.empty() || !std::filesystem::exists(python)) {
    std::printf("  SKIP  python bulunamadi (%s)\n", python.c_str());
    return kSkip;
  }

  std::vector<std::int32_t> tokens(static_cast<std::size_t>(c.batch) * static_cast<std::size_t>(c.time));
  std::uint64_t state = c.seed ^ 0x9E3779B97F4A7C15ULL;
  for (auto& token : tokens) {
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    token = static_cast<std::int32_t>((state >> 33) % static_cast<std::uint64_t>(c.vocab));
  }

  const Model net(c);
  const auto start = std::chrono::steady_clock::now();
  const Variable loss = net.loss(tokens);
  loss.backward();
  const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  std::printf("  info  nexus_loss=%.8f fwd+bwd=%.2fs\n", static_cast<double>(loss.values()[0]), seconds);

  const std::filesystem::path dir =
      std::filesystem::temp_directory_path() / ("nexus_parity_" + c.name + "_" + std::to_string(c.batch) + "x" +
                                                std::to_string(c.time));
  std::filesystem::create_directories(dir);
  {
    std::ofstream weights(dir / "weights.bin", std::ios::binary);
    std::ofstream grads(dir / "nexus_grads.bin", std::ios::binary);
    for (const Param& p : net.params()) {
      write_floats(weights, p.value.values());
      write_floats(grads, p.value.grad().values());
    }
  }
  {
    std::ofstream manifest(dir / "manifest.json", std::ios::binary);
    manifest << "{\n  \"config\": \"" << json_escape(std::filesystem::absolute(config_path).string()) << "\",\n";
    manifest << "  \"batch\": " << c.batch << ",\n  \"time\": " << c.time << ",\n";
    char loss_text[64];
    std::snprintf(loss_text, sizeof(loss_text), "%.9g", static_cast<double>(loss.values()[0]));  // FP32 tam hassasiyet
    manifest << "  \"nexus_loss\": " << loss_text << ",\n";
    manifest << "  \"mask_value\": -1e9,\n  \"ln_eps\": 1e-5,\n";
    manifest << "  \"tokens\": [";
    for (std::size_t i = 0; i < tokens.size(); ++i) {
      manifest << (i ? "," : "") << tokens[i];
    }
    manifest << "],\n  \"params\": [\n";
    for (std::size_t i = 0; i < net.params().size(); ++i) {
      const Param& p = net.params()[i];
      manifest << "    {\"name\": \"" << p.name << "\", \"shape\": [";
      for (std::size_t d = 0; d < p.value.shape().dims.size(); ++d) {
        manifest << (d ? "," : "") << p.value.shape().dims[d];
      }
      manifest << "]}" << (i + 1 < net.params().size() ? "," : "") << "\n";
    }
    manifest << "  ]\n}\n";
  }

  const std::string script_dir = NEXUS_PARITY_SCRIPTS;
  const std::string q = "\"";
  const int ref_status = run_status(q + python + q + " -I " + q + script_dir + "/reference_torch.py" + q + " " + q +
                                    dir.string() + q);
  if (ref_status == kSkip) {
    std::printf("  SKIP  PyTorch kurulu degil: parity SKIPPED\n");
    return kSkip;
  }
  if (ref_status != 0) {
    std::printf("  FAIL  reference_torch.py basarisiz (kod %d)\n", ref_status);
    return 1;
  }
  const bool compared = run_command(q + python + q + " -I " + q + script_dir + "/compare.py" + q + " " + q +
                                    dir.string() + q + " --loss-rtol 1e-5 --grad-rtol 1e-4");
  std::printf("  %s  Nexus vs PyTorch FP32 parity (%s)\n", compared ? "PASS" : "FAIL", c.name.c_str());
  std::printf("  info  rapor: %s\n", (dir / "parity_report.json").string().c_str());
  return compared ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  try {
    return run(argc, argv);
  } catch (const std::exception& error) {
    std::printf("  FAIL  beklenmeyen istisna: %s\n", error.what());
    return 1;
  }
}
