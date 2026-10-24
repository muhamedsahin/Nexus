#pragma once

/**
 * @file autograd.hpp
 * @brief Genel, tensör düzeyinde ters-mod autograd motoru — FAZ 2.
 *
 * PyTorch'un Node/Function + Edge + AccumulateGrad tasariminin sadelestirilmis
 * bir karsiligi: her turevlenebilir islem bir `Node` uretir, ciktinin `grad_fn`
 * bu dugume isaret eder, `next_edges` gradyana ihtiyac duyan girisleri tutar.
 * `backward()` bagimlilik sayaci + hazir kuyrugu ile topolojik sirada calisir;
 * yaprak tensörde gradyan `AccumulateGrad` ile `+=` biriktirilir (FAZ 9 icin
 * dogal). Kaydedilmis tensörler sürüm sayaciyla denetlenir: geri geçiste
 * yerinde degistirilmisse acik hata verilir.
 *
 * Kapsam:
 *  - Mod denetimi: `NoGradGuard`, `EnableGradGuard`, `set_grad_enabled`.
 *  - Kancalar: `Variable::register_hook`, `register_post_accumulate_hook`.
 *  - Op kümesi: add/mul/neg/sub/relu/gelu/silu/sigmoid/tanh/exp/log/pow,
 *    sum/mean, reshape/transpose, matmul, softmax, cross_entropy, dropout,
 *    add_bias (son boyutta yayin).
 *  - `saved_tensors_hooks` (FAZ 7 offload/sikistirma icin uzanti noktasi).
 *
 * Kapsam disi (FAZ 2'de yalniz altyapi, tam destek istege bagli): cift
 * backward (yuksek mertebeden türev) ve ileri-mod AD.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace nexus {
namespace autograd {

class Variable;
class Node;

/// Girdi gradyan kenari: bir dugumun hangi ciktisina gradyan akacagi.
struct Edge {
  std::shared_ptr<Node> function;
  std::uint32_t output_nr = 0;
};

/// Boyut (rank 0 = skaler; bos dims -> numel 1).
struct Shape {
  std::vector<std::int64_t> dims;

  [[nodiscard]] std::size_t rank() const noexcept { return dims.size(); }
  [[nodiscard]] std::int64_t numel() const noexcept {
    std::int64_t n = 1;
    for (const std::int64_t d : dims) {
      n *= d;
    }
    return n;
  }
  [[nodiscard]] bool empty() const noexcept { return numel() == 0; }
};

// ---- Grad modu --------------------------------------------------------------
[[nodiscard]] bool grad_enabled() noexcept;
void set_grad_enabled(bool enabled) noexcept;

/// Kapsam boyunca gradyan kaydi kapalidir (ileri geçiste Node uretilmez).
class NoGradGuard {
 public:
  NoGradGuard() noexcept;
  ~NoGradGuard();
  NoGradGuard(const NoGradGuard&) = delete;
  NoGradGuard& operator=(const NoGradGuard&) = delete;

 private:
  bool previous_;
};

/// Kapsam boyunca gradyan kaydi zorla aciktir.
class EnableGradGuard {
 public:
  EnableGradGuard() noexcept;
  ~EnableGradGuard();
  EnableGradGuard(const EnableGradGuard&) = delete;
  EnableGradGuard& operator=(const EnableGradGuard&) = delete;

 private:
  bool previous_;
};

// ---- Anomali modu -----------------------------------------------------------
/// Acikken: her dugum ileri geçis baglamini kaydeder; geri geçiste NaN/Inf
/// gradyan ureten dugum adiyla ve baglamiyla `std::runtime_error` firlatir.
[[nodiscard]] bool anomaly_mode() noexcept;
void set_anomaly_mode(bool enabled) noexcept;

class AnomalyModeGuard {
 public:
  explicit AnomalyModeGuard(bool enabled = true) noexcept;
  ~AnomalyModeGuard();
  AnomalyModeGuard(const AnomalyModeGuard&) = delete;
  AnomalyModeGuard& operator=(const AnomalyModeGuard&) = delete;

 private:
  bool previous_;
};

// ---- Node (PyTorch Function) -------------------------------------------------
class Node : public std::enable_shared_from_this<Node> {
 public:
  virtual ~Node() = default;

  /// grad_outputs -> grad_inputs (next_edges.size() adet).
  virtual std::vector<Variable> apply(std::vector<Variable>&& grad_outputs) = 0;

  std::vector<Edge> next_edges;
  std::uint64_t sequence_nr = 0;
  std::string name = "Node";
  /// Bu dugumun urettigi (yaprak olmayan) ciktinin ust-verisi: ara tensör
  /// kancalari ve retain_grad icin. Zayif: cikti -> meta -> grad_fn dongusu olmasin.
  std::weak_ptr<struct AutogradMeta> output_meta;
  /// Anomali modunda ileri geçisi kuran opun adi/baglami (geri geçis hatasinda raporlanir).
  std::string forward_context;
  /// retain_graph=false ile bir kez geri yurutulup kenarlari silindi mi.
  bool released = false;
};

/// Yeni dugume artan sira numarasi verir (topolojik sira, anomali raporu).
[[nodiscard]] std::uint64_t next_sequence_nr() noexcept;

/// Tensörün autograd ust-verisi (PyTorch AutogradMeta karsiligi).
struct AutogradMeta {
  bool requires_grad = false;
  bool is_leaf = false;
  bool retains_grad = false;
  std::uint64_t version = 0;
  std::shared_ptr<Node> grad_fn;
  std::shared_ptr<Variable> grad;
  std::vector<std::pair<std::uint64_t, std::function<Variable(const Variable&)>>> grad_hooks;
  std::vector<std::function<void()>> post_accumulate_hooks;
  std::uint64_t next_hook_id = 1;
  std::string name;
};

/// Yaprak tensörün gradyan biriktiricisi: grad += gelen gradyan.
class AccumulateGrad : public Node {
 public:
  explicit AccumulateGrad(std::weak_ptr<AutogradMeta> meta) : meta_(std::move(meta)) { name = "AccumulateGrad"; }
  std::vector<Variable> apply(std::vector<Variable>&& grad_outputs) override;

 private:
  std::weak_ptr<AutogradMeta> meta_;  // zayif: Variable->meta->grad_fn dongusunu kirmak icin
};

// ---- Kaydedilmis degisken ve kancalar -----------------------------------------
using SavedPackFn = std::function<std::shared_ptr<void>(const Variable&)>;
using SavedUnpackFn = std::function<Variable(std::shared_ptr<void>&&)>;

/// FAZ 7 offload/sikistirma icin genel kayit/geri-yukleme kancalari.
void set_saved_tensors_hooks(SavedPackFn pack, SavedUnpackFn unpack);
void clear_saved_tensors_hooks();

/// Kapsam boyunca pack/unpack kancalarini kurar, cikista oncekileri geri koyar
/// (PyTorch `torch.autograd.graph.saved_tensors_hooks` karsiligi).
class SavedTensorsHooksGuard {
 public:
  SavedTensorsHooksGuard(SavedPackFn pack, SavedUnpackFn unpack);
  ~SavedTensorsHooksGuard();
  SavedTensorsHooksGuard(const SavedTensorsHooksGuard&) = delete;
  SavedTensorsHooksGuard& operator=(const SavedTensorsHooksGuard&) = delete;

 private:
  SavedPackFn previous_pack_;
  SavedUnpackFn previous_unpack_;
};

/// Geri geçiste kullanmak uzere kaydedilmis degisken + sürüm denetimi.
///
/// Kanca yokken degiskenin kendisi (veriyi paylasarak) tutulur. Kanca varken
/// yalnizca pack ciktisi tutulur; ozgun veriye guclu referans kalmaz, boylece
/// offload/sikistirma gercekten bellek kazandirir. Her iki yolda da sürüm
/// denetimi ozgun tensörün ust-verisi uzerinden yapilir.
class SavedTensor {
 public:
  SavedTensor() = default;

  /// Sürüm degisti ise hata; aksi halde kaydedilen degiskeni (gerekirse unpack ile) dondurur.
  [[nodiscard]] Variable unpack(const char* op_name) const;
  [[nodiscard]] bool is_packed() const noexcept { return static_cast<bool>(unpack_fn_); }

 private:
  friend SavedTensor save_variable(const Variable& value);

  std::shared_ptr<const Variable> value_;
  std::shared_ptr<void> packed_;
  SavedUnpackFn unpack_fn_;
  std::weak_ptr<AutogradMeta> meta_;
  bool had_meta_ = false;
  std::uint64_t version_ = 0;
  Shape shape_;
};

/// Ileri geçiste kaydetme (kanca varsa paketler).
[[nodiscard]] SavedTensor save_variable(const Variable& value);
/// Geri geçiste okuma (kanca varsa geri yukler).
[[nodiscard]] Variable restore_variable(const SavedTensor& saved, const char* op_name);

/// Otograd tensörü (PyTorch Variable/Tensor karsiligi).
class Variable {
 public:
  Variable() = default;

  static Variable make(std::vector<float> values, Shape shape, bool requires_grad = false);
  static Variable zeros(Shape shape, bool requires_grad = true);
  static Variable ones(Shape shape, bool requires_grad = true);
  static Variable full(Shape shape, float value, bool requires_grad = true);
  static Variable randn(Shape shape, std::uint64_t seed, bool requires_grad = true);

  /// Operasyonlarin ciktiyi sürüm sayacini artirmadan kurmasi icin.
  static Variable from_data(std::shared_ptr<std::vector<float>> data, Shape shape,
                            std::shared_ptr<AutogradMeta> meta);

  [[nodiscard]] bool has_data() const noexcept { return static_cast<bool>(data_); }
  [[nodiscard]] const std::vector<float>& values() const noexcept;
  [[nodiscard]] std::vector<float>& mutable_values();  // sürümü artirir
  [[nodiscard]] const Shape& shape() const noexcept;
  [[nodiscard]] std::int64_t numel() const noexcept;
  [[nodiscard]] float operator[](std::size_t index) const;
  [[nodiscard]] float& at(std::size_t index);  // sürümü artirir

  [[nodiscard]] bool requires_grad() const noexcept;
  [[nodiscard]] bool is_leaf() const noexcept;
  [[nodiscard]] std::uint64_t version() const noexcept;
  [[nodiscard]] const std::shared_ptr<Node>& grad_fn() const noexcept;
  [[nodiscard]] Variable& grad() const;
  void set_grad(Variable gradient);
  void zero_grad();
  void retain_grad();

  void backward() const;
  void backward(const Variable& gradient, bool retain_graph = false) const;

  /// Grad kancasi; donen gradyan bu kancadan gecer. Kimlik dondurur.
  std::uint64_t register_hook(std::function<Variable(const Variable&)> hook);
  bool remove_hook(std::uint64_t hook_id);
  void register_post_accumulate_hook(std::function<void()> hook);

  // Yerinde degisiklikler (sürüm artirir; yaprak olmayan icin hata).
  void add_(const Variable& other);
  void mul_(float scalar);
  void fill_(float value);

  /// Grafikten kopuk, ayni veriyi paylasan gorunum (PyTorch detach).
  [[nodiscard]] Variable detach() const;
  /// Ust-veri (motor ve kayit icin); bos olabilir.
  [[nodiscard]] const std::shared_ptr<AutogradMeta>& meta() const noexcept { return meta_; }

 private:
  friend class AccumulateGrad;
  friend void backward(const std::vector<Variable>& roots, const std::vector<Variable>& grad_roots, bool retain_graph,
                       bool create_graph);

  std::shared_ptr<std::vector<float>> data_;
  Shape shape_;
  std::shared_ptr<AutogradMeta> meta_;
};

/// Op ciktisini kurar: dugume kenarlari, sira numarasini, cikti ust-verisini
/// (ara kancalar/retain_grad) ve anomali modunda ileri geçis baglamini baglar.
/// Ozel dugumler (CustomFunction, fused) bu tek yoldan cikti uretir.
[[nodiscard]] Variable make_op_output(std::vector<float> values, const Shape& shape, std::shared_ptr<Node> node,
                                      std::vector<Edge> edges);

// ---- Operasyonlar ----------------------------------------------------------------
Variable add(const Variable& a, const Variable& b);
Variable sub(const Variable& a, const Variable& b);
Variable mul(const Variable& a, const Variable& b);
Variable div(const Variable& a, const Variable& b);
Variable neg(const Variable& a);
Variable add_scalar(const Variable& a, float scalar);
Variable mul_scalar(const Variable& a, float scalar);
Variable add_bias(const Variable& x, const Variable& bias);
Variable relu(const Variable& a);
Variable gelu(const Variable& a);
Variable silu(const Variable& a);
Variable sigmoid(const Variable& a);
Variable tanh(const Variable& a);
Variable exp(const Variable& a);
Variable log(const Variable& a);
Variable pow(const Variable& a, float exponent);
Variable sum(const Variable& a);
Variable mean(const Variable& a);
Variable reshape(const Variable& a, Shape shape);
Variable transpose(const Variable& a);  // 2D
Variable matmul(const Variable& a, const Variable& b);  // 2D [m,k] x [k,n]
Variable softmax(const Variable& a);  // son boyut ([N,C])
Variable cross_entropy(const Variable& logits, const std::vector<std::int64_t>& targets);
Variable dropout(const Variable& a, float p, bool training, std::uint64_t seed);
Variable log_softmax(const Variable& a);  // son boyut ([N,C])
/// weight [V,D], indices N -> [N,D]. padding_idx satiri gradyan almaz (-1: yok).
Variable embedding(const Variable& weight, const std::vector<std::int64_t>& indices, std::int64_t padding_idx = -1);
/// 2D dilim: dim 0 (satir) veya 1 (sutun), [start, start+length).
Variable narrow(const Variable& a, int dim, std::int64_t start, std::int64_t length);
/// 2D birlestirme: dim 0 veya 1.
Variable cat(const std::vector<Variable>& parts, int dim);

// ---- CustomFunction (PyTorch autograd.Function karsiligi) ------------------------
/// grad_output -> giris basina bir gradyan (requires_grad=false girisler icin bos Variable olabilir).
using CustomBackwardFn = std::function<std::vector<Variable>(const Variable& grad_output)>;

/// Elle yazilmis ileri/geri cifti tek dugum olarak teybe girer. `output` cagiranin
/// grafiksiz hesapladigi ileri sonuctur; `backward_fn` geri gecişte cagrilir.
/// Elle optimize edilmis klasik katman backward'lari boylece kaybolmaz.
Variable custom_function(std::string name, const std::vector<Variable>& inputs, std::vector<float> output,
                         Shape output_shape, CustomBackwardFn backward_fn);

// ---- Aktivasyon yeniden hesaplama (FAZ 7.1) ------------------------------------------
struct CheckpointOptions {
  /// Nexus'ta rastgelelik (dropout_philox, dropout, randn) acik (seed, offset) alir: global RNG
  /// durumu yoktur, yeniden calistirma ayni maskeyi kendiliginden uretir. Bayrak sozlesmeyi
  /// belgeler; false verilmesi desteklenmez (sessiz farkli maske yok).
  bool preserve_rng = true;
};

/// Non-reentrant checkpoint (PyTorch use_reentrant=False karsiligi). Ileri geciste fn normal
/// grafla calisir ama kaydedilen tensörler saved-tensor kancasiyla yalniz sira numarasi olarak
/// tutulur; geri geciste ilk unpack fn'i girislerden yeniden calistirir ve kaydedilenleri ayni
/// sirayla doldurur. Ic ice checkpoint, kismi backward ve retain_graph ile uyumlu. fn
/// belirlenimci olmali (ayni sirada ayni op'lar).
Variable checkpoint(const std::function<Variable(const std::vector<Variable>&)>& fn, const std::vector<Variable>& inputs,
                    const CheckpointOptions& options = {});

/// Olcum: kapsam boyunca save_variable ile kaydedilen bayt (kanca zincirine eklenir).
class SavedBytesCounter {
 public:
  SavedBytesCounter();
  ~SavedBytesCounter();
  SavedBytesCounter(const SavedBytesCounter&) = delete;
  SavedBytesCounter& operator=(const SavedBytesCounter&) = delete;
  [[nodiscard]] std::size_t bytes() const noexcept { return *bytes_; }

 private:
  std::shared_ptr<std::size_t> bytes_;
  std::unique_ptr<SavedTensorsHooksGuard> guard_;
};

// ---- Motor ------------------------------------------------------------------------
void backward(const std::vector<Variable>& roots, const std::vector<Variable>& grad_roots, bool retain_graph,
              bool create_graph = false);
void backward(const Variable& root, const Variable& gradient, bool retain_graph = false, bool create_graph = false);

// ---- Gradcheck ------------------------------------------------------------------
struct GradcheckResult {
  bool ok = false;
  double max_abs_error = 0.0;
  double max_rel_error = 0.0;
  std::size_t worst_input = 0;
  std::size_t worst_index = 0;
};

/// Merkezi sonlu farkla analitik gradyani dogrular. `fn` skaler bir kayip
/// dondurmelidir. Fark hesabi FP64 biriktirilir; tensör verisi FP32 oldugundan
/// varsayilan toleranslar FP32'ye gore secilmistir (|a-n| <= atol + rtol*|n|).
[[nodiscard]] GradcheckResult gradcheck(const std::function<Variable(const std::vector<Variable>&)>& fn,
                                        const std::vector<Variable>& inputs, float eps = 1.0e-3f,
                                        double atol = 2.0e-3, double rtol = 2.0e-2);

}  // namespace autograd
}  // namespace nexus
