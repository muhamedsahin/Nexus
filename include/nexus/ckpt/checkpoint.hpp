#pragma once

/**
 * @file checkpoint.hpp
 * @brief FAZ 15: dayanikli kayit — TrainerState, NXS3 arsivi, atomik kayit dizini, manifest + SHA-256, son-N saklama,
 *        rank basina parcali dosya (tek rankta toplama yok), eszamansiz (cift tamponlu) yazici.
 *
 * NXS3 arsivi (tek dosya):
 *   "NXS3\0\0\0\0" | u32 surum (3) | u32 girdi sayisi | girdi: u32 ad uzunlugu, ad, u64 bayt, veri | 32 bayt SHA-256
 *   (sondaki ozet kendinden onceki her baytin ozetidir; okumada dogrulanir, uyusmazsa acik hata).
 * Kayit dizini (adim basina):
 *   <dir>/step-000000123/manifest.json  {"format":"NXS3","version":3,"step":123,"world":4,"layout":"...",
 *                                         "files":[{"name":"rank0.nxs3","bytes":...,"sha256":"..."}, ...],"state":{...}}
 *   <dir>/step-000000123/rank<r>.nxs3    girdiler: "state" (TrainerState, bu rank'in veri imleci), "backend" (arka uc yuku)
 *   <dir>/latest                         son tamamlanan adim (atomik yazilir)
 * Manifest'teki sha256 = arsiv icerigi ozeti (dosyanin son 32 bayti); yazmada tek SHA gecisi, yuklemede
 * boyut + ozet manifest ile karsilastirilir ve icerik decode_archive'da yeniden ozetlenip dogrulanir.
 * Atomiklik: her rank kendi dosyasini <dir>/.tmp-step-.../rank<r>.nxs3.part'a yazar, flush eder, ".nxs3"e yeniden adlandirir
 * ve yaninda rank<r>.done (sha + bayt) birakir. Rank 0 tum .done dosyalarini gorunce manifest.json'i yazar ve gecici dizini
 * TEK yeniden adlandirmayla step-... yapar; sonra latest guncellenir ve en eski kayitlar silinir (keep). Yarim kalan kayit
 * (cokme) .tmp-... olarak kalir; yukleme onu gormez, rank 0 bir sonraki baslangicta temizler.
 * Rank'ler arasi tamamlama dosya sistemi uzerinden yapilir (kollektif cagri yok): eszamansiz yazici egitim
 * kollektifleriyle sira yarisina girmez. Varsayim: tum rank'ler ayni dizini gorur (tek dugum ya da paylasimli FS).
 * Eszamansiz kip: save() yalniz yukun sahipligini alir (cagiran bayt kopyasini zaten yapmistir: ilk tampon) ve
 * arka plan is parcacigina verir; ayni anda en cok bir kayit yaziliyor + bir kayit bekliyor (cift tampon). Ucuncu
 * save() oncekinin bitmesini bekler.
 * Cokme enjeksiyonu (test): NEXUS_CKPT_CRASH = "rank_file" | "before_manifest" | "before_rename" — o noktada
 * surec std::_Exit(86) ile olur (yalniz NEXUS_CKPT_CRASH_RANK rank'inde ve NEXUS_CKPT_CRASH_STEP adiminda, verilmisse).
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include <array>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace nexus {
namespace ckpt {

class CkptError : public std::runtime_error {
 public:
  explicit CkptError(const std::string& what) : std::runtime_error("Nexus ckpt: " + what) {}
};

/// SHA-256: SHA-NI varsa donanim yolu (allow_hw), yoksa NexusData skaler. Iki yol ayni ozeti verir.
[[nodiscard]] std::array<std::uint8_t, 32> sha256(const void* data, std::size_t size, bool allow_hw = true);
[[nodiscard]] const char* sha256_backend();
[[nodiscard]] std::string sha256_hex(const void* data, std::size_t size);

struct Entry {
  std::string name;
  std::vector<unsigned char> data;
};
inline constexpr std::uint32_t kArchiveVersion = 3;
[[nodiscard]] std::vector<unsigned char> encode_archive(const std::vector<Entry>& entries);
/// Surum ve SHA-256 dogrulanir; `what` hata iletisinde kaynagi adlandirir.
[[nodiscard]] std::vector<Entry> decode_archive(const std::vector<unsigned char>& bytes, const std::string& what);
[[nodiscard]] const Entry& find_entry(const std::vector<Entry>& entries, const std::string& name);

/// path.part'a yazar, flush eder, path'e yeniden adlandirir (varsa uzerine).
void write_file_atomic(const std::filesystem::path& path, const void* data, std::size_t size);
[[nodiscard]] std::vector<unsigned char> read_file(const std::filesystem::path& path);

/// Philox tarzi sayac tabanli RNG durumu: maske/ornek = f(seed, offset, cagri). offset = tamamlanan mikro adim sayisi.
struct RngState {
  std::uint64_t seed = 0;
  std::uint64_t offset = 0;
};

struct TrainerState {
  std::int64_t step = 0;                 // tamamlanan optimizer adimi
  std::int64_t epoch = 0;
  std::uint64_t global_sample_index = 0; // tum rank'lerde tuketilen satir sayisi
  std::string train_cursor;              // bu rank'in veri kaynagi durumu
  std::string val_cursor;
  double lr_factor = 1.0;                // NaN geri donusunun LR carpani (zamanlayici adimi = step)
  std::int64_t schedule_step = 0;
  double loss_scale = 1.0;
  RngState rng;
  std::int32_t accum_counter = 0;        // kayit adim sinirinda: 0
  std::int32_t accum_steps = 1;
  std::int32_t micro_batch = 1;
  std::int32_t seq_len = 2;
  std::int64_t global_batch_tokens = 0;
  std::int32_t rollbacks = 0;
  std::int32_t world = 1;
  std::int32_t rank = 0;
  [[nodiscard]] std::vector<unsigned char> encode() const;
  [[nodiscard]] static TrainerState decode(const std::vector<unsigned char>& bytes);
  [[nodiscard]] std::string to_json() const;
};

struct CheckpointOptions {
  std::string dir;
  std::int64_t interval = 0;  // > 0: her interval adimda kayit
  int keep = 3;               // son-N tamamlanmis kayit tutulur (<= 0: hepsi)
  bool async = false;
  std::chrono::milliseconds timeout{std::chrono::minutes(5)};  // rank 0'in diger rank dosyalarini bekleme siniri
};

struct SaveStats {
  std::int64_t step = -1;
  double blocking_ms = 0.0;   // save() cagrisinin egitim is parcacigini tuttugu sure
  double write_ms = 0.0;      // kodlama + SHA + diske yazma + (rank 0) tamamlama
  std::uint64_t bytes = 0;    // bu rank'in dosyasi
  [[nodiscard]] double gb_per_s() const { return write_ms > 0.0 ? static_cast<double>(bytes) / (write_ms * 1e6) : 0.0; }
};

struct ManifestFile {
  std::string name;
  std::uint64_t bytes = 0;
  std::string sha256;
};
struct Manifest {
  std::string format;
  std::uint32_t version = 0;
  std::int64_t step = 0;
  int world = 1;
  std::string layout;
  std::vector<ManifestFile> files;
};
[[nodiscard]] Manifest read_manifest(const std::filesystem::path& step_dir);

struct LoadedCheckpoint {
  std::filesystem::path dir;
  Manifest manifest;
  bool resharded = false;                        // kayit dunyasi != bu dunya
  /// Ayni dunya: [bu rank]. Farkli dunya: eski rank sirasiyla tum yukler (yeniden parcalama icin).
  std::vector<std::vector<unsigned char>> payloads;
  std::vector<TrainerState> states;              // payloads ile ayni sira
};

class CheckpointManager {
 public:
  /// Rank 0 kurulumda yarim kalmis .tmp-* dizinlerini siler: egitim kollektiflerinden ONCE kurulmali.
  CheckpointManager(CheckpointOptions options, int rank, int world);
  ~CheckpointManager();
  CheckpointManager(const CheckpointManager&) = delete;
  CheckpointManager& operator=(const CheckpointManager&) = delete;

  [[nodiscard]] bool due(std::int64_t step) const noexcept {
    return options_.interval > 0 && step > 0 && step % options_.interval == 0;
  }
  /// Kollektif degil, ama her rank ayni adimda cagirmali. Senkron kipte kayit tamamlaninca doner.
  void save(std::int64_t step, const TrainerState& state, std::vector<unsigned char> payload, const std::string& layout);
  /// Bu rank'in bekleyen yazimlarini (rank 0: tamamlamayi da) bekler; arka plan hatasini yeniden firlatir.
  void wait();
  /// En yeni tamamlanmis kayit (manifest + okunan dosyalarin SHA'si dogrulanir). Yoksa nullopt.
  [[nodiscard]] std::optional<LoadedCheckpoint> load_latest() const;
  [[nodiscard]] std::vector<std::int64_t> committed_steps() const;
  [[nodiscard]] SaveStats last_stats() const;
  [[nodiscard]] const CheckpointOptions& options() const noexcept { return options_; }

  [[nodiscard]] static std::filesystem::path step_dir(const std::filesystem::path& root, std::int64_t step);

 private:
  struct Job {
    std::int64_t step = 0;
    TrainerState state;
    std::vector<unsigned char> payload;
    std::string layout;
    double blocking_ms = 0.0;
  };
  void write_job(Job& job);
  void worker();

  CheckpointOptions options_;
  int rank_ = 0;
  int world_ = 1;
  std::filesystem::path root_;
  mutable std::mutex mu_;
  std::condition_variable cv_;
  std::optional<Job> pending_;
  bool busy_ = false;
  bool stop_ = false;
  std::string error_;
  SaveStats stats_;
  std::thread thread_;
};

}  // namespace ckpt
}  // namespace nexus
