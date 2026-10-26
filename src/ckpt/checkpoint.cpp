/**
 * @file checkpoint.cpp
 * @brief FAZ 15: NXS3 arsivi, atomik kayit dizini, manifest + SHA-256, son-N saklama, eszamansiz yazici.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/ckpt/checkpoint.hpp"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>

namespace fs = std::filesystem;

namespace nexus {
namespace ckpt {

namespace {

using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

constexpr char kMagic[8] = {'N', 'X', 'S', '3', 0, 0, 0, 0};

std::string env(const char* key) {
#ifdef _MSC_VER
  char* v = nullptr;
  std::size_t n = 0;
  if (_dupenv_s(&v, &n, key) == 0 && v != nullptr) {
    std::string s = v;
    std::free(v);
    return s;
  }
  return {};
#else
  const char* v = std::getenv(key);
  return v ? v : "";
#endif
}

bool crash_armed(const char* stage, int rank, std::int64_t step) {
  const std::string st = env("NEXUS_CKPT_CRASH_STEP");
  if (!st.empty() && std::stoll(st) != step) return false;
  if (env("NEXUS_CKPT_CRASH") != stage) return false;
  const std::string r = env("NEXUS_CKPT_CRASH_RANK");
  return r.empty() || std::atoi(r.c_str()) == rank;
}

void crash_point(const char* stage, int rank, std::int64_t step) {
  if (!crash_armed(stage, rank, step)) return;
  std::fprintf(stderr, "nexus ckpt: NEXUS_CKPT_CRASH=%s -> surec olduruluyor (rank %d)\n", stage, rank);
  std::fflush(stderr);
  std::_Exit(86);
}

struct Writer {
  std::vector<unsigned char> out;
  template <typename T>
  void pod(const T& v) {
    const auto* p = reinterpret_cast<const unsigned char*>(&v);
    out.insert(out.end(), p, p + sizeof(T));
  }
  void str(const std::string& s) {
    pod<std::uint64_t>(s.size());
    out.insert(out.end(), s.begin(), s.end());
  }
};

struct Reader {
  const std::vector<unsigned char>& in;
  std::size_t pos = 0;
  std::string what;
  void need(std::size_t n) const {
    if (pos + n > in.size()) throw CkptError(what + ": kisa veri (bozuk ya da kesik)");
  }
  template <typename T>
  T pod() {
    need(sizeof(T));
    T v;
    std::memcpy(&v, in.data() + pos, sizeof(T));
    pos += sizeof(T);
    return v;
  }
  std::string str() {
    const auto n = static_cast<std::size_t>(pod<std::uint64_t>());
    need(n);
    std::string s(reinterpret_cast<const char*>(in.data() + pos), n);
    pos += n;
    return s;
  }
};

std::string json_escape(const std::string& s) {
  std::string o;
  for (char c : s) {
    switch (c) {
      case '"': o += "\\\""; break;
      case '\\': o += "\\\\"; break;
      case '\n': o += "\\n"; break;
      case '\r': o += "\\r"; break;
      case '\t': o += "\\t"; break;
      default:
        if (static_cast<unsigned char>(c) < 0x20) {
          char buf[8];
          std::snprintf(buf, sizeof(buf), "\\u%04x", c);
          o += buf;
        } else {
          o += c;
        }
    }
  }
  return o;
}

// ---- kucuk JSON okuyucu (yalniz manifest icin) ------------------------------------------
struct JValue {
  enum Kind { Null, Bool, Num, Str, Arr, Obj } kind = Null;
  double num = 0.0;
  bool b = false;
  std::string str;
  std::vector<JValue> arr;
  std::map<std::string, JValue> obj;
  const JValue& at(const std::string& k) const {
    auto it = obj.find(k);
    if (kind != Obj || it == obj.end()) throw CkptError("manifest: eksik alan '" + k + "'");
    return it->second;
  }
};

class JParser {
 public:
  explicit JParser(const std::string& s) : s_(s) {}
  JValue parse() {
    JValue v = value();
    ws();
    if (i_ != s_.size()) fail("fazla veri");
    return v;
  }

 private:
  [[noreturn]] void fail(const char* why) const { throw CkptError(std::string("manifest JSON: ") + why); }
  void ws() {
    while (i_ < s_.size() && std::isspace(static_cast<unsigned char>(s_[i_]))) ++i_;
  }
  char peek() {
    ws();
    if (i_ >= s_.size()) fail("beklenmedik son");
    return s_[i_];
  }
  JValue value() {
    const char c = peek();
    JValue v;
    if (c == '{') {
      v.kind = JValue::Obj;
      ++i_;
      if (peek() == '}') {
        ++i_;
        return v;
      }
      for (;;) {
        if (peek() != '"') fail("anahtar bekleniyordu");
        std::string k = string();
        if (peek() != ':') fail("':' bekleniyordu");
        ++i_;
        v.obj[k] = value();
        const char d = peek();
        ++i_;
        if (d == '}') break;
        if (d != ',') fail("',' bekleniyordu");
      }
    } else if (c == '[') {
      v.kind = JValue::Arr;
      ++i_;
      if (peek() == ']') {
        ++i_;
        return v;
      }
      for (;;) {
        v.arr.push_back(value());
        const char d = peek();
        ++i_;
        if (d == ']') break;
        if (d != ',') fail("',' bekleniyordu");
      }
    } else if (c == '"') {
      v.kind = JValue::Str;
      v.str = string();
    } else if (s_.compare(i_, 4, "true") == 0) {
      v.kind = JValue::Bool;
      v.b = true;
      i_ += 4;
    } else if (s_.compare(i_, 5, "false") == 0) {
      v.kind = JValue::Bool;
      i_ += 5;
    } else if (s_.compare(i_, 4, "null") == 0) {
      i_ += 4;
    } else {
      v.kind = JValue::Num;
      const char* b = s_.c_str() + i_;
      char* e = nullptr;
      v.num = std::strtod(b, &e);
      if (e == b) fail("deger bekleniyordu");
      i_ += static_cast<std::size_t>(e - b);
    }
    return v;
  }
  std::string string() {
    ++i_;  // "
    std::string o;
    while (i_ < s_.size() && s_[i_] != '"') {
      char c = s_[i_++];
      if (c == '\\') {
        if (i_ >= s_.size()) fail("kacis");
        const char e = s_[i_++];
        switch (e) {
          case 'n': o += '\n'; break;
          case 'r': o += '\r'; break;
          case 't': o += '\t'; break;
          case 'u': {
            if (i_ + 4 > s_.size()) fail("\\u");
            o += static_cast<char>(std::strtol(s_.substr(i_, 4).c_str(), nullptr, 16));
            i_ += 4;
            break;
          }
          default: o += e;
        }
      } else {
        o += c;
      }
    }
    if (i_ >= s_.size()) fail("kapanmamis dize");
    ++i_;
    return o;
  }
  const std::string& s_;
  std::size_t i_ = 0;
};

/// Arsivin son 32 bayti (icerigin SHA-256'si) onaltilik: manifest bunu kaydeder.
std::string trailer_hex(const std::vector<unsigned char>& bytes) {
  static const char* hex = "0123456789abcdef";
  std::string o(64, '0');
  const unsigned char* d = bytes.data() + bytes.size() - 32;
  for (std::size_t i = 0; i < 32; ++i) {
    o[2 * i] = hex[d[i] >> 4];
    o[2 * i + 1] = hex[d[i] & 15];
  }
  return o;
}

std::string rank_file(int r) { return "rank" + std::to_string(r) + ".nxs3"; }

fs::path tmp_dir(const fs::path& root, std::int64_t step) {
  char buf[64];
  std::snprintf(buf, sizeof(buf), ".tmp-step-%09lld", static_cast<long long>(step));
  return root / buf;
}

bool parse_step_dir(const std::string& name, std::int64_t* step) {
  if (name.size() != 14 || name.compare(0, 5, "step-") != 0) return false;
  for (std::size_t i = 5; i < name.size(); ++i)
    if (!std::isdigit(static_cast<unsigned char>(name[i]))) return false;
  *step = std::stoll(name.substr(5));
  return true;
}

}  // namespace

// ------------------------------------------------------------------------------------------

std::string sha256_hex(const void* data, std::size_t size) {
  const auto d = sha256(data, size);
  static const char* hex = "0123456789abcdef";
  std::string o(64, '0');
  for (std::size_t i = 0; i < 32; ++i) {
    o[2 * i] = hex[d[i] >> 4];
    o[2 * i + 1] = hex[d[i] & 15];
  }
  return o;
}

std::vector<unsigned char> encode_archive(const std::vector<Entry>& entries) {
  Writer w;
  w.out.insert(w.out.end(), kMagic, kMagic + 8);
  w.pod<std::uint32_t>(kArchiveVersion);
  w.pod<std::uint32_t>(static_cast<std::uint32_t>(entries.size()));
  std::size_t total = w.out.size() + 32;
  for (const Entry& e : entries) total += 4 + e.name.size() + 8 + e.data.size();
  w.out.reserve(total);
  for (const Entry& e : entries) {
    w.pod<std::uint32_t>(static_cast<std::uint32_t>(e.name.size()));
    w.out.insert(w.out.end(), e.name.begin(), e.name.end());
    w.pod<std::uint64_t>(e.data.size());
    w.out.insert(w.out.end(), e.data.begin(), e.data.end());
  }
  const auto d = sha256(w.out.data(), w.out.size());
  w.out.insert(w.out.end(), d.begin(), d.end());
  return std::move(w.out);
}

std::vector<Entry> decode_archive(const std::vector<unsigned char>& bytes, const std::string& what) {
  if (bytes.size() < 16 + 32 || std::memcmp(bytes.data(), kMagic, 8) != 0) {
    throw CkptError(what + ": NXS3 arsivi degil");
  }
  {
    const auto d = sha256(bytes.data(), bytes.size() - 32);
    if (std::memcmp(d.data(), bytes.data() + bytes.size() - 32, 32) != 0) {
      throw CkptError(what + ": SHA-256 uyusmuyor (dosya bozuk)");
    }
  }
  Reader r{bytes, 8, what};
  const auto version = r.pod<std::uint32_t>();
  if (version != kArchiveVersion) {
    throw CkptError(what + ": desteklenmeyen arsiv surumu " + std::to_string(version) + " (beklenen 3)");
  }
  const auto n = r.pod<std::uint32_t>();
  std::vector<Entry> out(n);
  for (Entry& e : out) {
    const auto len = r.pod<std::uint32_t>();
    r.need(len);
    e.name.assign(reinterpret_cast<const char*>(bytes.data() + r.pos), len);
    r.pos += len;
    const auto sz = static_cast<std::size_t>(r.pod<std::uint64_t>());
    if (r.pos + sz > bytes.size() - 32) throw CkptError(what + ": girdi tasiyor");
    e.data.assign(bytes.begin() + static_cast<std::ptrdiff_t>(r.pos), bytes.begin() + static_cast<std::ptrdiff_t>(r.pos + sz));
    r.pos += sz;
  }
  if (r.pos != bytes.size() - 32) throw CkptError(what + ": arsiv sonunda fazla veri");
  return out;
}

const Entry& find_entry(const std::vector<Entry>& entries, const std::string& name) {
  for (const Entry& e : entries)
    if (e.name == name) return e;
  throw CkptError("arsivde '" + name + "' girdisi yok");
}

void write_file_atomic(const fs::path& path, const void* data, std::size_t size) {
  fs::path part = path;
  part += ".part";
  {
    std::ofstream f(part, std::ios::binary | std::ios::trunc);
    if (!f) throw CkptError("yazilamadi: " + part.string());
    f.write(static_cast<const char*>(data), static_cast<std::streamsize>(size));
    f.flush();
    if (!f) throw CkptError("yazma hatasi: " + part.string());
  }
  std::error_code ec;
  fs::rename(part, path, ec);
  if (ec) throw CkptError("yeniden adlandirilamadi: " + part.string() + " -> " + path.string() + " (" + ec.message() + ")");
}

std::vector<unsigned char> read_file(const fs::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw CkptError("acilamadi: " + path.string());
  in.seekg(0, std::ios::end);
  const auto n = static_cast<std::size_t>(in.tellg());
  in.seekg(0);
  std::vector<unsigned char> v(n);
  in.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(n));
  if (!in) throw CkptError("okunamadi: " + path.string());
  return v;
}

// ------------------------------------------------------------------------------------------

std::vector<unsigned char> TrainerState::encode() const {
  Writer w;
  w.pod<std::uint32_t>(1);  // TrainerState sema surumu
  w.pod(step);
  w.pod(epoch);
  w.pod(global_sample_index);
  w.str(train_cursor);
  w.str(val_cursor);
  w.pod(lr_factor);
  w.pod(schedule_step);
  w.pod(loss_scale);
  w.pod(rng.seed);
  w.pod(rng.offset);
  w.pod(accum_counter);
  w.pod(accum_steps);
  w.pod(micro_batch);
  w.pod(seq_len);
  w.pod(global_batch_tokens);
  w.pod(rollbacks);
  w.pod(world);
  w.pod(rank);
  return std::move(w.out);
}

TrainerState TrainerState::decode(const std::vector<unsigned char>& bytes) {
  Reader r{bytes, 0, "TrainerState"};
  if (r.pod<std::uint32_t>() != 1) throw CkptError("TrainerState: desteklenmeyen sema surumu");
  TrainerState s;
  s.step = r.pod<std::int64_t>();
  s.epoch = r.pod<std::int64_t>();
  s.global_sample_index = r.pod<std::uint64_t>();
  s.train_cursor = r.str();
  s.val_cursor = r.str();
  s.lr_factor = r.pod<double>();
  s.schedule_step = r.pod<std::int64_t>();
  s.loss_scale = r.pod<double>();
  s.rng.seed = r.pod<std::uint64_t>();
  s.rng.offset = r.pod<std::uint64_t>();
  s.accum_counter = r.pod<std::int32_t>();
  s.accum_steps = r.pod<std::int32_t>();
  s.micro_batch = r.pod<std::int32_t>();
  s.seq_len = r.pod<std::int32_t>();
  s.global_batch_tokens = r.pod<std::int64_t>();
  s.rollbacks = r.pod<std::int32_t>();
  s.world = r.pod<std::int32_t>();
  s.rank = r.pod<std::int32_t>();
  if (r.pos != bytes.size()) throw CkptError("TrainerState: fazla veri");
  return s;
}

std::string TrainerState::to_json() const {
  std::ostringstream o;
  o.precision(17);
  o << "{\"step\":" << step << ",\"epoch\":" << epoch << ",\"global_sample_index\":" << global_sample_index
    << ",\"lr_factor\":" << lr_factor << ",\"schedule_step\":" << schedule_step << ",\"loss_scale\":" << loss_scale
    << ",\"rng\":{\"seed\":" << rng.seed << ",\"offset\":" << rng.offset << "},\"accum_counter\":" << accum_counter
    << ",\"accum_steps\":" << accum_steps << ",\"micro_batch\":" << micro_batch << ",\"seq_len\":" << seq_len
    << ",\"global_batch_tokens\":" << global_batch_tokens << ",\"rollbacks\":" << rollbacks << ",\"world\":" << world
    << ",\"train_cursor_rank0\":\"" << json_escape(train_cursor) << "\"}";
  return o.str();
}

Manifest read_manifest(const fs::path& step_dir) {
  const std::vector<unsigned char> raw = read_file(step_dir / "manifest.json");
  const JValue j = JParser(std::string(raw.begin(), raw.end())).parse();
  Manifest m;
  m.format = j.at("format").str;
  m.version = static_cast<std::uint32_t>(j.at("version").num);
  if (m.format != "NXS3" || m.version != kArchiveVersion) {
    throw CkptError(step_dir.string() + ": desteklenmeyen kayit bicimi " + m.format + " v" + std::to_string(m.version));
  }
  m.step = static_cast<std::int64_t>(j.at("step").num);
  m.world = static_cast<int>(j.at("world").num);
  m.layout = j.at("layout").str;
  for (const JValue& f : j.at("files").arr) {
    m.files.push_back(ManifestFile{f.at("name").str, static_cast<std::uint64_t>(f.at("bytes").num), f.at("sha256").str});
  }
  if (static_cast<int>(m.files.size()) != m.world) throw CkptError(step_dir.string() + ": manifest dosya sayisi != world");
  return m;
}

// ------------------------------------------------------------------------------------------

fs::path CheckpointManager::step_dir(const fs::path& root, std::int64_t step) {
  char buf[64];
  std::snprintf(buf, sizeof(buf), "step-%09lld", static_cast<long long>(step));
  return root / buf;
}

CheckpointManager::CheckpointManager(CheckpointOptions options, int rank, int world)
    : options_(std::move(options)), rank_(rank), world_(world), root_(options_.dir) {
  if (options_.dir.empty()) throw CkptError("CheckpointOptions::dir bos");
  std::error_code ec;
  fs::create_directories(root_, ec);
  if (ec) throw CkptError("dizin olusturulamadi: " + root_.string());
  if (rank_ == 0) {
    for (const auto& e : fs::directory_iterator(root_)) {
      const std::string n = e.path().filename().string();
      if (n.compare(0, 5, ".tmp-") == 0) fs::remove_all(e.path(), ec);
    }
  }
  if (options_.async) thread_ = std::thread([this] { worker(); });
}

CheckpointManager::~CheckpointManager() {
  {
    std::lock_guard<std::mutex> lock(mu_);
    stop_ = true;
  }
  cv_.notify_all();
  if (thread_.joinable()) thread_.join();
}

void CheckpointManager::save(std::int64_t step, const TrainerState& state, std::vector<unsigned char> payload,
                             const std::string& layout) {
  const auto t0 = Clock::now();
  Job job{step, state, std::move(payload), layout, 0.0};
  if (!options_.async) {
    job.blocking_ms = 0.0;
    write_job(job);
    std::lock_guard<std::mutex> lock(mu_);
    stats_.blocking_ms = ms_since(t0);
    return;
  }
  std::unique_lock<std::mutex> lock(mu_);
  cv_.wait(lock, [&] { return !pending_.has_value() || !error_.empty(); });  // cift tampon: bekleyen yuva bos olmali
  if (!error_.empty()) throw CkptError("onceki eszamansiz kayit basarisiz: " + error_);
  job.blocking_ms = ms_since(t0);
  pending_ = std::move(job);
  lock.unlock();
  cv_.notify_all();
}

void CheckpointManager::worker() {
  for (;;) {
    Job job;
    {
      std::unique_lock<std::mutex> lock(mu_);
      cv_.wait(lock, [&] { return stop_ || pending_.has_value(); });
      if (!pending_.has_value()) return;  // stop_ ve is yok
      job = std::move(*pending_);
      pending_.reset();
      busy_ = true;
    }
    cv_.notify_all();
    try {
      write_job(job);
    } catch (const std::exception& e) {
      std::lock_guard<std::mutex> lock(mu_);
      error_ = e.what();
    }
    {
      std::lock_guard<std::mutex> lock(mu_);
      busy_ = false;
    }
    cv_.notify_all();
  }
}

void CheckpointManager::wait() {
  std::unique_lock<std::mutex> lock(mu_);
  cv_.wait(lock, [&] { return (!pending_.has_value() && !busy_) || !error_.empty(); });
  if (!error_.empty()) throw CkptError("eszamansiz kayit basarisiz: " + error_);
}

void CheckpointManager::write_job(Job& job) {
  const auto t0 = Clock::now();
  const fs::path tmp = tmp_dir(root_, job.step);
  const fs::path final_dir = step_dir(root_, job.step);
  std::error_code ec;
  fs::create_directories(tmp, ec);
  if (ec && !fs::exists(tmp)) throw CkptError("gecici dizin olusturulamadi: " + tmp.string());
  std::vector<Entry> entries;
  entries.push_back(Entry{"state", job.state.encode()});
  entries.push_back(Entry{"backend", std::move(job.payload)});
  std::vector<unsigned char> bytes = encode_archive(entries);
  entries.clear();
  const std::string sha = trailer_hex(bytes);  // arsiv sonundaki ozet: ikinci SHA gecisi yok
  const fs::path file = tmp / rank_file(rank_);
  if (crash_armed("rank_file", rank_, job.step)) {
    // yarim dosya birak (gercek cokme gibi): .part'in ilk yarisi
    fs::path part = file;
    part += ".part";
    std::ofstream f(part, std::ios::binary | std::ios::trunc);
    f.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size() / 2));
    f.close();
    crash_point("rank_file", rank_, job.step);
  }
  write_file_atomic(file, bytes.data(), bytes.size());
  const std::string done = sha + " " + std::to_string(bytes.size());
  write_file_atomic(tmp / ("rank" + std::to_string(rank_) + ".done"), done.data(), done.size());
  const std::uint64_t nbytes = bytes.size();
  bytes.clear();
  bytes.shrink_to_fit();

  const auto deadline = Clock::now() + options_.timeout;
  if (rank_ == 0) {
    std::vector<ManifestFile> files(static_cast<std::size_t>(world_));
    for (int r = 0; r < world_; ++r) {
      const fs::path d = tmp / ("rank" + std::to_string(r) + ".done");
      while (!fs::exists(d)) {
        if (Clock::now() > deadline) {
          throw CkptError("adim " + std::to_string(job.step) + ": rank " + std::to_string(r) + " dosyasi zaman asimina kadar gelmedi");
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
      const std::vector<unsigned char> raw = read_file(d);
      std::istringstream in(std::string(raw.begin(), raw.end()));
      ManifestFile mf;
      mf.name = rank_file(r);
      in >> mf.sha256 >> mf.bytes;
      files[static_cast<std::size_t>(r)] = mf;
    }
    std::ostringstream m;
    m << "{\"format\":\"NXS3\",\"version\":" << kArchiveVersion << ",\"step\":" << job.step << ",\"world\":" << world_
      << ",\"layout\":\"" << json_escape(job.layout) << "\",\"files\":[";
    for (int r = 0; r < world_; ++r) {
      const ManifestFile& f = files[static_cast<std::size_t>(r)];
      m << (r ? "," : "") << "{\"name\":\"" << f.name << "\",\"bytes\":" << f.bytes << ",\"sha256\":\"" << f.sha256 << "\"}";
    }
    m << "],\"state\":" << job.state.to_json() << "}\n";
    crash_point("before_manifest", rank_, job.step);
    const std::string ms = m.str();
    write_file_atomic(tmp / "manifest.json", ms.data(), ms.size());
    for (int r = 0; r < world_; ++r) fs::remove(tmp / ("rank" + std::to_string(r) + ".done"), ec);
    crash_point("before_rename", rank_, job.step);
    if (fs::exists(final_dir)) fs::remove_all(final_dir, ec);  // ayni adim yeniden kaydediliyor
    fs::rename(tmp, final_dir, ec);
    if (ec) throw CkptError("kayit tamamlanamadi (yeniden adlandirma): " + ec.message());
    const std::string latest = std::to_string(job.step);
    write_file_atomic(root_ / "latest", latest.data(), latest.size());
    if (options_.keep > 0) {
      const std::vector<std::int64_t> steps = committed_steps();
      for (std::size_t i = 0; i + static_cast<std::size_t>(options_.keep) < steps.size(); ++i) {
        fs::remove_all(step_dir(root_, steps[i]), ec);
      }
    }
  } else {
    // diger rank'ler: tamamlanmayi bekle (senkron anlam: save dondugunde kayit kalici)
    while (!fs::exists(final_dir / "manifest.json")) {
      if (Clock::now() > deadline) {
        throw CkptError("adim " + std::to_string(job.step) + ": rank 0 kaydi zaman asimina kadar tamamlamadi");
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }
  std::lock_guard<std::mutex> lock(mu_);
  stats_.step = job.step;
  stats_.write_ms = ms_since(t0);
  stats_.bytes = nbytes;
  stats_.blocking_ms = job.blocking_ms;
}

std::vector<std::int64_t> CheckpointManager::committed_steps() const {
  std::vector<std::int64_t> out;
  std::error_code ec;
  if (!fs::exists(root_, ec)) return out;
  for (const auto& e : fs::directory_iterator(root_, ec)) {
    std::int64_t s = 0;
    if (e.is_directory() && parse_step_dir(e.path().filename().string(), &s) && fs::exists(e.path() / "manifest.json")) {
      out.push_back(s);
    }
  }
  std::sort(out.begin(), out.end());
  return out;
}

std::optional<LoadedCheckpoint> CheckpointManager::load_latest() const {
  const std::vector<std::int64_t> steps = committed_steps();
  if (steps.empty()) return std::nullopt;
  LoadedCheckpoint lc;
  lc.dir = step_dir(root_, steps.back());
  lc.manifest = read_manifest(lc.dir);
  lc.resharded = lc.manifest.world != world_;
  auto load_one = [&](int r) {
    const ManifestFile& mf = lc.manifest.files[static_cast<std::size_t>(r)];
    const std::vector<unsigned char> bytes = read_file(lc.dir / mf.name);
    if (bytes.size() != mf.bytes || bytes.size() < 32 || trailer_hex(bytes) != mf.sha256) {  // icerik ozeti decode_archive dogrular
      throw CkptError(lc.dir.string() + "/" + mf.name + ": manifest ile uyusmuyor (boyut ya da SHA-256)");
    }
    std::vector<Entry> entries = decode_archive(bytes, mf.name);
    lc.states.push_back(TrainerState::decode(find_entry(entries, "state").data));
    for (Entry& e : entries)
      if (e.name == "backend") lc.payloads.push_back(std::move(e.data));
    if (lc.payloads.size() != lc.states.size()) throw CkptError(mf.name + ": 'backend' girdisi yok");
  };
  if (!lc.resharded) {
    load_one(rank_);
  } else {
    for (int r = 0; r < lc.manifest.world; ++r) load_one(r);
  }
  return lc;
}

SaveStats CheckpointManager::last_stats() const {
  std::lock_guard<std::mutex> lock(mu_);
  return stats_;
}

}  // namespace ckpt
}  // namespace nexus
