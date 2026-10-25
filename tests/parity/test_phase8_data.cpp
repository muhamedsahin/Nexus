/**
 * @file test_phase8_data.cpp
 * @brief FAZ 8 testleri: BPE kodlama (tiktoken ile bayt esitligi), cozme + akisli UTF-8, BPE
 *        egitimi (HF tokenizers ile ayni birlesim sirasi), NXTK parca gidis-donus + SHA, ornekleyici
 *        devami, paketleme maskesi, rank'lar arasi cakismazlik.
 *
 * Python (tiktoken/tokenizers) yoksa ilgili modlar 77 (SKIPPED) doner.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexusdata/text/bpe.hpp"
#include "nexusdata/text/token_shard.hpp"
#include "nexus_test.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <psapi.h>
#endif

namespace nt = nexusdata::text;
namespace fs = std::filesystem;

namespace {

constexpr int kSkip = 77;

int run_status(const std::string& command) {
#if defined(_WIN32)
  return std::system(("\"" + command + "\"").c_str());
#else
  return std::system(command.c_str());
#endif
}

bool have_python() {
  const std::string py = NEXUS_PARITY_PYTHON;
  return !py.empty() && fs::exists(py);
}

std::string python(const std::string& args) {
  return std::string("\"") + NEXUS_PARITY_PYTHON + "\" -I \"" + NEXUS_PARITY_SCRIPTS + "/reference_tokenizer.py\" " + args;
}

fs::path scratch(const std::string& name) {
  const fs::path d = fs::temp_directory_path() / ("nexus_faz8_" + name);
  fs::create_directories(d);
  return d;
}

void report(const std::string& name, double err, double tol) {
  std::printf("  info  %-58s hata=%.3e esik=%.1e\n", name.c_str(), err, tol);
  nexus_test::expect(err <= tol, name);
}

// ---- Deterministik cok dilli belge ureteci --------------------------------------------------
struct Gen {
  std::uint64_t s;
  explicit Gen(std::uint64_t seed) : s(seed) {}
  std::uint64_t next() {
    s = s * 6364136223846793005ULL + 1442695040888963407ULL;
    return s >> 33;
  }
  std::size_t pick(std::size_t n) { return static_cast<std::size_t>(next() % n); }
};

const std::vector<std::string>& atoms() {
  static const std::vector<std::string> a = {
      "the", "The", "quick", "brown", "fox", "jumps", "over", "lazy", "dog", "Hello", "world", "naïve", "café",
      "e\xCC\x81t\xC3\xA9",  // birlesik aksan (e + U+0301)
      "İstanbul", "ığüşöç", "ĞÜŞİÖÇ", "Москва", "привет", "Ελλάδα", "日本語", "中文字符", "한국어", "עברית", "العربية",
      "हिन्दी",  // Devanagari: harf + \p{M} isaretleri
      "ไทย", "😀", "👍🏽", "🇹🇷", "∑", "≈", "→", "…", "—", "«", "»", "“", "”", "€", "$", "%", "&", "#", "@", "_", "-", "+",
      "=", "/", "\\", "|", "<", ">", "{", "}", "[", "]", "(", ")", ",", ".", ";", ":", "!", "?", "*", "^", "~", "`", "\"",
      "'", "'s", "'S", "'t", "'re", "'RE", "'Ve", "'m", "'ll", "'LL", "'d", "'D", "'\xC5\xBF",  // 'ſ (uzun s)
      "can't", "I'm", "you're", "''", "'''s", "0", "7", "42", "1234", "1234567", "3.14159", "1,000,000", "٣٤٥",
      "０１２", "²", "½", "Ⅻ", "x²", "v1.2.3", "0x1F", "\xEF\xBB\xBF", "\xE2\x80\x8B",  // BOM, sifir genislik
      "http://example.com/a?b=c&d=e", "foo_bar", "camelCase", "SNAKE_CASE", "<|endoftext|>", "</s>", "ﬁ", "Å", "ß"};
  return a;
}

const std::vector<std::string>& spaces() {
  static const std::vector<std::string> s = {" ", " ", " ", "  ", "   ", "\t", "\n", "\n\n", "\r\n", " \n", "\n ",
                                             "  \n\n  ", "\xC2\xA0", "\xE3\x80\x80", "\xE2\x80\x83", "\x0B", "\x0C",
                                             "\xC2\x85", "\xE2\x80\xA8", " \t ", ""};
  return s;
}

std::vector<std::string> make_docs(int n, std::uint64_t seed) {
  Gen g(seed);
  std::vector<std::string> docs;
  for (int d = 0; d < n; ++d) {
    std::string doc;
    const std::size_t len = g.pick(5) == 0 ? g.pick(8) : 10 + g.pick(300);
    if (g.pick(10) == 0) doc += spaces()[g.pick(spaces().size())];  // bosluk/satir sonuyla baslayan
    for (std::size_t k = 0; k < len; ++k) {
      doc += atoms()[g.pick(atoms().size())];
      if (g.pick(4) != 0) doc += spaces()[g.pick(spaces().size())];
    }
    if (g.pick(4) == 0) doc += spaces()[g.pick(spaces().size())];  // sonda bosluk (\s++$ kurali)
    docs.push_back(std::move(doc));
  }
  return docs;
}

void write_docs(const fs::path& path, const std::vector<std::string>& docs) {
  std::ofstream out(path, std::ios::binary);
  const auto n = static_cast<std::uint32_t>(docs.size());
  out.write(reinterpret_cast<const char*>(&n), 4);
  for (const std::string& d : docs) {
    const auto len = static_cast<std::uint32_t>(d.size());
    out.write(reinterpret_cast<const char*>(&len), 4);
    out.write(d.data(), static_cast<std::streamsize>(d.size()));
  }
}

std::vector<std::vector<std::uint32_t>> read_ref(const fs::path& path, std::size_t n) {
  std::ifstream in(path, std::ios::binary);
  std::vector<std::vector<std::uint32_t>> out(n);
  for (std::size_t i = 0; i < n; ++i) {
    std::uint32_t len = 0;
    in.read(reinterpret_cast<char*>(&len), 4);
    out[i].resize(len);
    in.read(reinterpret_cast<char*>(out[i].data()), static_cast<std::streamsize>(len * 4));
  }
  return out;
}

fs::path vocab_dir() {
  const fs::path d = scratch("vocab");
  if (!fs::exists(d / "gpt2.tiktoken") || !fs::exists(d / "cl100k.tiktoken")) {
    if (run_status(python("export \"" + d.string() + "\"")) != 0) {
      throw std::runtime_error("tiktoken sozlukleri disa aktarilamadi");
    }
  }
  return d;
}

nt::ByteBpe load(const std::string& name) {
  const fs::path d = vocab_dir();
  std::vector<std::pair<std::string, std::uint32_t>> specials;
  if (name == "gpt2") {
    specials = {{"<|endoftext|>", 50256}};
  } else {
    specials = {{"<|endoftext|>", 100257}, {"<|fim_prefix|>", 100258}, {"<|fim_middle|>", 100259},
                {"<|fim_suffix|>", 100260}, {"<|endofprompt|>", 100276}};
  }
  return nt::ByteBpe::load_tiktoken((d / (name + ".tiktoken")).string(),
                                    name == "gpt2" ? nt::Pretokenizer::Gpt2 : nt::Pretokenizer::Cl100k, specials);
}

// Uzun on-token parcalari (> 256 bayt: yigin tabanli birlesim yolu): bosluk/sembol/harf/rakam
// dizileri ve az harfli rastgele kelimeler (cok birlesim adimi).
std::vector<std::string> make_long_docs(int n, std::uint64_t seed) {
  Gen g(seed);
  const char* runs[] = {" ", "=", "-", "*", "a", "0", "\t", "\xC4\x9F", "\xE3\x81\x82", "ab", "the", "\n"};
  const char* letters[] = {"a", "b", "e", "t", "h", "s", "n", "\xC3\xBC", "\xC5\x9F"};
  std::vector<std::string> docs;
  for (int d = 0; d < n; ++d) {
    std::string doc = "x ";
    const std::size_t len = 200 + g.pick(4000);
    if (d % 2 == 0) {
      const std::string r = runs[g.pick(std::size(runs))];
      while (doc.size() < len) doc += r;
    } else {
      while (doc.size() < len) doc += letters[g.pick(std::size(letters))];
    }
    doc += " y";
    docs.push_back(std::move(doc));
  }
  return docs;
}

int compare_with_tiktoken(const fs::path& dir, const std::vector<std::string>& docs);

int test_tiktoken() {
  nexus_test::section("test_bpe_encode_matches_tiktoken: 10k uretilmis + 400 uzun parcali belge, GPT-2 ve cl100k");
  if (!have_python()) {
    std::printf("  SKIP  python yok\n");
    return kSkip;
  }
  std::vector<std::string> docs = make_docs(10000, 0xFA28);
  for (auto& d : make_long_docs(400, 0x1046)) docs.push_back(std::move(d));
  return compare_with_tiktoken(scratch("encode"), docs);
}

int compare_with_tiktoken(const fs::path& dir, const std::vector<std::string>& docs) {
  write_docs(dir / "docs.bin", docs);
  std::size_t bytes = 0;
  for (const auto& d : docs) bytes += d.size();
  std::printf("  info  %zu belge, %.2f MB, Unicode tablolari %s\n", docs.size(), bytes / 1e6, nt::unicode_version());
  for (const char* name : {"gpt2", "cl100k"}) {
    if (run_status(python(std::string("encode \"") + dir.string() + "\" " + name)) != 0) {
      std::printf("  FAIL  tiktoken kodlamasi calismadi\n");
      return 1;
    }
    const auto ref = read_ref(dir / (std::string("ref_") + name + ".bin"), docs.size());
    const nt::ByteBpe bpe = load(name);
    std::size_t bad = 0;
    std::size_t first_bad = docs.size();
    std::size_t tokens = 0;
    for (std::size_t i = 0; i < docs.size(); ++i) {
      const auto got = bpe.encode(docs[i]);
      tokens += ref[i].size();
      if (got != ref[i]) {
        if (bad == 0) first_bad = i;
        ++bad;
      }
    }
    std::printf("  info  %s: %zu token, farkli belge %zu\n", name, tokens, bad);
    if (bad != 0) {
      const auto got = bpe.encode(docs[first_bad]);
      std::size_t k = 0;
      while (k < got.size() && k < ref[first_bad].size() && got[k] == ref[first_bad][k]) ++k;
      std::string ctx;
      for (std::size_t j = k; j < std::min(ref[first_bad].size(), k + 4); ++j) ctx += "[" + bpe.token_bytes(ref[first_bad][j]) + "]";
      std::string mine;
      for (std::size_t j = k; j < std::min(got.size(), k + 4); ++j) mine += "[" + bpe.token_bytes(got[j]) + "]";
      std::printf("  info  ilk fark belge %zu, token %zu: tiktoken %s / nexus %s\n", first_bad, k, ctx.c_str(), mine.c_str());
    }
    nexus_test::expect(bad == 0, std::string(name) + ": tum belgeler tiktoken ile token-token ayni");
    // Paralel kodlama ayni sonucu verir.
    std::vector<std::string_view> views(docs.begin(), docs.end());
    const auto par = bpe.encode_batch(views, 8);
    nexus_test::expect(par == ref, std::string(name) + ": encode_batch (8 is parcacigi) ayni");
  }
  return 0;
}

void test_roundtrip() {
  nexus_test::section("test_decode_roundtrip + akisli UTF-8");
  const nt::ByteBpe bpe = load("cl100k");
  const std::vector<std::string> docs = make_docs(2000, 0xDEC0);
  bool ok = true;
  bool stream_ok = true;
  std::size_t partial_seen = 0;
  for (const std::string& d : docs) {
    const auto ids = bpe.encode(d);
    ok = ok && bpe.decode(ids) == d;
    nt::StreamDecoder sd(bpe);
    std::string acc;
    for (const std::uint32_t id : ids) {
      const std::string piece = sd.push(id);
      // her parca kendi basina gecerli UTF-8 olmali (yarim dizi yok)
      std::size_t i = 0;
      while (i < piece.size()) {
        const auto b = static_cast<unsigned char>(piece[i]);
        const std::size_t need = b < 0x80 ? 1 : (b >> 5) == 6 ? 2 : (b >> 4) == 14 ? 3 : (b >> 3) == 30 ? 4 : 0;
        if (need == 0 || i + need > piece.size()) {
          stream_ok = false;
          break;
        }
        i += need;
      }
      if (piece.size() < bpe.token_bytes(id).size()) ++partial_seen;
      acc += piece;
    }
    acc += sd.flush();
    stream_ok = stream_ok && acc == d;
  }
  nexus_test::expect(ok, "decode(encode(x)) == x (2000 belge)");
  nexus_test::expect(stream_ok, "akisli cozme: her parca tam UTF-8, birlesimi ayni metin");
  std::printf("  info  akisli cozmede yarim UTF-8 bekletilen token: %zu\n", partial_seen);
  nexus_test::expect(partial_seen > 0, "akisli cozme yarim baytlari gercekten tamponladi (test kapsami)");
}

int test_train() {
  nexus_test::section("test_bpe_train_matches_hf: ayni birlesim sirasi ve sozluk");
  if (!have_python()) {
    std::printf("  SKIP  python yok\n");
    return kSkip;
  }
  const fs::path dir = scratch("train");
  // Kucuk korpus: uretilmis belgeler + depo metinleri (gercek dagilim).
  std::vector<std::string> corpus = make_docs(3000, 0x7EA1);
  for (const char* f : {"README.md", "promt.txt"}) {
    const fs::path p = fs::path(NEXUS_SOURCE_DIR) / f;
    std::ifstream in(p, std::ios::binary);
    std::stringstream ss;
    ss << in.rdbuf();
    std::string text = ss.str();
    for (std::size_t i = 0; i < text.size(); i += 4000) corpus.push_back(text.substr(i, 4000));
  }
  // Belge sinirlarinda UTF-8 bolunmesin diye gecersiz parcalari ele.
  corpus.erase(std::remove_if(corpus.begin(), corpus.end(), [](const std::string& s) {
                 std::size_t i = 0;
                 while (i < s.size()) {
                   const auto b = static_cast<unsigned char>(s[i]);
                   const std::size_t need = b < 0x80 ? 1 : (b >> 5) == 6 ? 2 : (b >> 4) == 14 ? 3 : (b >> 3) == 30 ? 4 : 0;
                   if (need == 0 || i + need > s.size()) return true;
                   i += need;
                 }
                 return false;
               }),
               corpus.end());
  write_docs(dir / "corpus.bin", corpus);
  const std::uint32_t vocab = 2000;
  if (run_status(python("train \"" + dir.string() + "\" " + std::to_string(vocab))) != 0) {
    std::printf("  FAIL  HF egitimi calismadi\n");
    return 1;
  }
  std::vector<std::string_view> views(corpus.begin(), corpus.end());
  nt::BpeTrainOptions opt;
  opt.vocab_size = vocab;
  opt.special_tokens = {"<|endoftext|>"};
  const auto t0 = std::chrono::steady_clock::now();
  const nt::BpeTrainResult r = nt::train_bpe(views, opt);
  const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  std::vector<std::string> hf_merges;
  {
    std::ifstream in(dir / "hf_merges.txt", std::ios::binary);
    std::string line;
    while (std::getline(in, line)) {
      hf_merges.push_back(line);
    }
  }
  std::vector<std::string> mine;
  for (const auto& [a, b] : r.merges) mine.push_back(a + " " + b);
  std::size_t k = 0;
  while (k < mine.size() && k < hf_merges.size() && mine[k] == hf_merges[k]) ++k;
  std::printf("  info  %zu belge, %llu farkli kelime | birlesim: nexus %zu, HF %zu, ortak onek %zu | nexus %.2f s\n",
              corpus.size(), static_cast<unsigned long long>(r.words), mine.size(), hf_merges.size(), k, secs);
  if (k < hf_merges.size()) {
    std::printf("  info  ilk fark #%zu: HF '%s' / nexus '%s'\n", k, hf_merges[k].c_str(),
                k < mine.size() ? mine[k].c_str() : "-");
  }
  nexus_test::expect(mine == hf_merges, "birlesim listesi HF ile birebir ayni");
  std::vector<std::string> hf_vocab;
  {
    std::ifstream in(dir / "hf_vocab.txt", std::ios::binary);
    std::string line;
    while (std::getline(in, line)) {
      const std::size_t sp = line.find(' ');
      hf_vocab.push_back(line.substr(sp + 1));
    }
  }
  nexus_test::expect(hf_vocab == r.vocab, "sozluk (id -> token) HF ile birebir ayni");
  // Egitilmis tokenizer kodlar ve geri cozer.
  bool rt = true;
  for (std::size_t i = 0; i < 200 && i < corpus.size(); ++i) {
    rt = rt && r.tokenizer.decode(r.tokenizer.encode(corpus[i])) == corpus[i];
  }
  nexus_test::expect(rt, "egitilmis tokenizer gidis-donus");
  return 0;
}

// ---- NXTK parcalari ve ornekleyici ----------------------------------------------------------
// Uretilmis belgelerden kucuk parcalarla (birden cok parca) bir veri kumesi.
nt::Manifest build_dataset(const fs::path& dir, int docs, std::uint64_t seed, std::uint64_t shard_tokens) {
  fs::remove_all(dir);
  fs::create_directories(dir);
  const nt::ByteBpe bpe = load("gpt2");
  const std::vector<std::string> text = make_docs(docs, seed);
  nt::Manifest m;
  m.tokenizer_name = "gpt2";
  m.tokenizer_digest = nt::to_hex(bpe.digest());
  m.vocab_size = bpe.vocab_size();
  m.eos = 50256;
  std::unique_ptr<nt::ShardWriter> w;
  std::string name;
  auto close = [&] {
    if (!w) return;
    const auto sha = w->finish();
    m.shards.push_back({name, w->tokens(), w->docs(), nt::to_hex(sha)});
    w.reset();
  };
  for (const std::string& t : text) {
    if (!w) {
      name = "shard_" + std::to_string(m.shards.size()) + ".nxtk";
      w = std::make_unique<nt::ShardWriter>((dir / name).string(), bpe.vocab_size(), bpe.digest(), 50256);
    }
    const auto ids = bpe.encode(t);
    w->add_document(ids.data(), ids.size());
    if (w->tokens() >= shard_tokens) close();
  }
  close();
  m.save((dir / "manifest.json").string());
  return nt::Manifest::load((dir / "manifest.json").string());
}

void test_shard() {
  nexus_test::section("test_shard_roundtrip + SHA");
  const fs::path dir = scratch("shard");
  const nt::Manifest m = build_dataset(dir, 1500, 0x5AD, 60000);
  const nt::ByteBpe bpe = load("gpt2");
  const std::vector<std::string> text = make_docs(1500, 0x5AD);
  std::size_t doc = 0;
  bool same = true;
  for (const auto& s : m.shards) {
    const nt::ShardReader r((dir / s.path).string());
    same = same && r.header().dtype_bytes == 2 && r.num_tokens() == s.tokens && r.num_docs() == s.docs;
    for (std::uint64_t d = 0; d < r.num_docs() && same; ++d, ++doc) {
      auto ids = bpe.encode(text[doc]);
      ids.push_back(50256);
      const std::uint64_t a = r.doc_offset(d);
      const std::uint64_t b = r.doc_offset(d + 1);
      same = same && b - a == ids.size();
      for (std::uint64_t i = a; i < b && same; ++i) same = same && r.token(i) == ids[i - a];
    }
  }
  std::printf("  info  %zu parca, %llu token, %zu belge\n", m.shards.size(), static_cast<unsigned long long>(m.total_tokens()), doc);
  nexus_test::expect(same && doc == text.size(), "tum belgeler token-token ve sinir-sinir geri okunur (EOS dahil)");
  nexus_test::expect(m.shards.size() > 3, "birden cok parca");
  nexus_test::expect(m.verify().empty(), "manifest SHA-256 dogrulamasi gecer");
  {
    std::fstream f(dir / m.shards[1].path, std::ios::binary | std::ios::in | std::ios::out);
    f.seekp(200);
    const char x = 0x5A;
    f.write(&x, 1);
  }
  nexus_test::expect(m.verify() == m.shards[1].path, "bozulmus parca SHA ile yakalanir");
}

std::uint64_t sample_hash(const nt::PackedSample& s) {
  std::uint64_t h = 1469598103934665603ULL;
  auto eat = [&](std::uint64_t v) { h = (h ^ v) * 1099511628211ULL; };
  for (const auto t : s.tokens) eat(static_cast<std::uint64_t>(t));
  for (const auto c : s.cu_seqlens) eat(static_cast<std::uint64_t>(c) + 0x100000000ULL);
  for (const auto m : s.loss_mask) eat(m);
  eat(s.source);
  return h;
}

int test_resume(const char* exe) {
  nexus_test::section("test_sampler_resume_exact: N ornek, durum kaydi, YENI SURECTE devam");
  const fs::path dir = scratch("resume");
  const nt::Manifest m = build_dataset(dir / "data", 200, 0x4E5, 20000);
  // Kesintisiz akis: 300 ornek (epoch sinirini gecer).
  nt::TokenStream full(m, 512, 99);
  std::vector<std::uint64_t> ref;
  nt::PackedSample s;
  std::string state_at_120;
  for (int i = 0; i < 300; ++i) {
    if (i == 120) state_at_120 = full.state().serialize();
    full.next(s);
    ref.push_back(sample_hash(s));
  }
  std::printf("  info  pencere/epoch %llu, durum@120 = %s (%zu bayt)\n",
              static_cast<unsigned long long>(full.windows_per_epoch()), state_at_120.c_str(), state_at_120.size());
  nexus_test::expect(full.windows_per_epoch() < 300, "test epoch sinirini kapsar");
  // Yeni surec: ayni manifest + kayitli durum -> kalan 180 ornegin ozetleri.
  const fs::path out = dir / "child.txt";
  const std::string cmd = std::string("\"") + exe + "\" resume_child \"" + (dir / "data" / "manifest.json").string() +
                          "\" " + state_at_120 + " \"" + out.string() + "\"";
  if (run_status(cmd) != 0) {
    std::printf("  FAIL  alt surec calismadi\n");
    return 1;
  }
  std::ifstream in(out);
  std::vector<std::uint64_t> child;
  std::uint64_t v = 0;
  while (in >> v) child.push_back(v);
  const std::vector<std::uint64_t> tail(ref.begin() + 120, ref.end());
  nexus_test::expect(child == tail, "yeni surecte devam: sonraki 180 ornek kesintisiz akisla birebir ayni");
  // Karisim da devam eder.
  auto make_mix = [&] {
    std::vector<std::unique_ptr<nt::TokenStream>> src;
    src.push_back(std::make_unique<nt::TokenStream>(m, 64, 1));
    src.push_back(std::make_unique<nt::TokenStream>(m, 64, 2));
    src.push_back(std::make_unique<nt::TokenStream>(m, 64, 3));
    return nt::MixtureStream(std::move(src), {0.7, 0.2, 0.1}, 7);
  };
  nt::MixtureStream a = make_mix();
  std::vector<std::uint64_t> ma;
  std::string mstate;
  std::size_t counts[3] = {0, 0, 0};
  for (int i = 0; i < 4000; ++i) {
    if (i == 2500) mstate = a.serialize_state();
    a.next(s);
    ++counts[s.source];
    ma.push_back(sample_hash(s));
  }
  nt::MixtureStream b = make_mix();
  b.restore_state(mstate);
  std::vector<std::uint64_t> mb;
  for (int i = 2500; i < 4000; ++i) {
    b.next(s);
    mb.push_back(sample_hash(s));
  }
  nexus_test::expect(mb == std::vector<std::uint64_t>(ma.begin() + 2500, ma.end()), "karisim durumu devam: birebir");
  std::printf("  info  karisim oranlari (hedef .70/.20/.10): %.3f / %.3f / %.3f\n", counts[0] / 4000.0, counts[1] / 4000.0,
              counts[2] / 4000.0);
  nexus_test::expect(std::abs(counts[0] / 4000.0 - 0.7) < 0.03 && std::abs(counts[2] / 4000.0 - 0.1) < 0.02,
                     "karisim agirliklari (+-3 sigma)");
  return 0;
}

int resume_child(int argc, char** argv) {
  if (argc < 5) return 2;
  const nt::Manifest m = nt::Manifest::load(argv[2]);
  nt::TokenStream st(m, 512, 99);
  st.restore(nt::SamplerState::deserialize(argv[3]));
  std::ofstream out(argv[4]);
  nt::PackedSample s;
  for (int i = 0; i < 180; ++i) {
    st.next(s);
    out << sample_hash(s) << "\n";
  }
  return 0;
}

void test_pack() {
  nexus_test::section("test_pack_mask: belge sinirinda cu_seqlens ve kayip maskesi");
  const fs::path dir = scratch("pack");
  const nt::Manifest m = build_dataset(dir, 800, 0x9AC, 1u << 30);
  const nt::ShardReader r((dir / m.shards[0].path).string());
  nt::TokenStream st(m, 96, 5);
  nt::PackedSample s;
  bool ok = true;
  std::size_t boundaries = 0;
  for (int i = 0; i < 2000 && ok; ++i) {
    st.next(s);
    const std::uint64_t start = s.window * 96;
    // Kaba kuvvet: hangi pozisyonlar yeni belge baslatir?
    std::vector<std::int32_t> cu{0};
    std::vector<std::uint8_t> mask(96, 1);
    for (std::uint64_t p = 1; p <= 96; ++p) {
      const bool doc_start = r.doc_of(start + p) != r.doc_of(start + p - 1);
      if (doc_start && p < 96) cu.push_back(static_cast<std::int32_t>(p));
      if (doc_start) mask[p - 1] = 0;
    }
    cu.push_back(96);
    boundaries += cu.size() - 2;
    ok = ok && cu == s.cu_seqlens && mask == s.loss_mask;
    // EOS'tan sonraki ilk token'in hedefi maskeli, EOS'un kendisi hedef olarak sayilir.
    for (std::size_t p = 0; p < 96 && ok; ++p) {
      if (s.tokens[p] == 50256 && p + 1 < 96 + 1) ok = ok && s.loss_mask[p] == 0;
    }
  }
  std::printf("  info  2000 pencere, %zu belge siniri\n", boundaries);
  nexus_test::expect(ok, "cu_seqlens ve kayip maskesi kaba kuvvetle ayni; EOS->sonraki belge tahmini maskeli");
  nexus_test::expect(boundaries > 100, "test belge sinirlarini kapsar");
  // En iyi uyum paketleme
  std::vector<std::uint64_t> lens;
  for (std::uint64_t d = 0; d < r.num_docs(); ++d) lens.push_back(r.doc_offset(d + 1) - r.doc_offset(d));
  const auto bins = nt::best_fit_pack(lens, 1024);
  std::vector<int> used(lens.size(), 0);
  bool fit = true;
  std::uint64_t filled = 0;
  for (const auto& b : bins) {
    std::uint64_t sum = 0;
    for (const auto d : b) {
      ++used[d];
      sum += lens[d];
    }
    fit = fit && (sum <= 1024 || b.size() == 1);
    filled += std::min<std::uint64_t>(sum, 1024);
  }
  const bool once = std::all_of(used.begin(), used.end(), [](int u) { return u == 1; });
  std::printf("  info  best-fit: %zu belge -> %zu kutu, doluluk %.1f%%\n", lens.size(), bins.size(),
              100.0 * filled / (bins.size() * 1024.0));
  nexus_test::expect(once && fit, "best-fit: her belge bir kez, kutular kapasite icinde, belge kesilmez");
}

void test_ranks() {
  nexus_test::section("test_no_overlap_across_ranks: 4 rank, bir epoch");
  const fs::path dir = scratch("ranks");
  const nt::Manifest m = build_dataset(dir, 1000, 0x2A4, 40000);
  const int world = 4;
  std::set<std::pair<std::uint64_t, std::uint64_t>> all;
  std::size_t total = 0;
  bool disjoint = true;
  std::uint64_t per_epoch = 0;
  for (int r = 0; r < world; ++r) {
    nt::TokenStream st(m, 100, 11, r, world);
    per_epoch = st.windows_per_epoch();
    nt::PackedSample s;
    const std::uint64_t mine = (per_epoch + static_cast<std::uint64_t>(world - 1 - r)) / world;
    for (std::uint64_t i = 0; i < mine; ++i) {
      st.next(s);
      disjoint = disjoint && all.insert({s.shard, s.window}).second;
      ++total;
    }
  }
  std::printf("  info  epoch basina %llu pencere, rank'lar toplam %zu\n", static_cast<unsigned long long>(per_epoch), total);
  nexus_test::expect(disjoint, "rank'lar arasi ortusme yok");
  nexus_test::expect(all.size() == per_epoch, "rank'larin birlesimi epoch'taki tum pencereler (her biri bir kez)");
  nt::FeistelPermutation p(1000003, 9);
  std::vector<char> hit(1000003, 0);
  bool perm = true;
  for (std::uint64_t i = 0; i < 1000003; ++i) {
    const std::uint64_t v = p(i);
    perm = perm && v < 1000003 && hit[v] == 0;
    hit[v] = 1;
  }
  nexus_test::expect(perm, "Feistel: n=1000003 icin gercek permutasyon");
}

// ---- Olcum modlari (ctest'te degil; rapor icin) ---------------------------------------------
std::vector<std::string> read_docs_file(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  std::uint32_t n = 0;
  in.read(reinterpret_cast<char*>(&n), 4);
  std::vector<std::string> docs(n);
  for (auto& d : docs) {
    std::uint32_t len = 0;
    in.read(reinterpret_cast<char*>(&len), 4);
    d.resize(len);
    in.read(d.data(), len);
  }
  return docs;
}

int verify_dir(const fs::path& dir) {
  nexus_test::section("verify_dir: " + dir.string() + " tiktoken ile");
  return compare_with_tiktoken(scratch("verify"), read_docs_file((dir / "docs.bin").string()));
}

int bench_encode(const std::string& path, const std::string& name) {
  nexus_test::section("bench_encode: " + name + " " + path);
  const std::vector<std::string> docs = read_docs_file(path);
  std::vector<std::string_view> views(docs.begin(), docs.end());
  std::size_t bytes = 0;
  for (const auto& d : docs) bytes += d.size();
  const nt::ByteBpe bpe = load(name);
  for (const int th : {1, 8, 16}) {
    (void)bpe.encode_batch(std::vector<std::string_view>(views.begin(), views.begin() + 200), th);
    double best = 1e30;
    std::size_t tokens = 0;
    for (int r = 0; r < 3; ++r) {
      const auto t0 = std::chrono::steady_clock::now();
      const auto out = bpe.encode_batch(views, th);
      best = std::min(best, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
      tokens = 0;
      for (const auto& o : out) tokens += o.size();
    }
    std::printf("  info  nexus %s %2d is parcacigi: %.2f MB/s (%zu belge, %.1f MB, %zu token)\n", name.c_str(), th,
                bytes / best / 1e6, docs.size(), bytes / 1e6, tokens);
  }
  return 0;
}

int bench_train(const std::string& dir, std::uint32_t vocab, int threads) {
  nexus_test::section("bench_train: " + dir + " vocab " + std::to_string(vocab));
  const std::vector<std::string> docs = read_docs_file((fs::path(dir) / "corpus.bin").string());
  std::vector<std::string_view> views(docs.begin(), docs.end());
  std::size_t bytes = 0;
  for (const auto& d : docs) bytes += d.size();
  nt::BpeTrainOptions opt;
  opt.vocab_size = vocab;
  opt.special_tokens = {"<|endoftext|>"};
  opt.threads = threads;
  const auto t0 = std::chrono::steady_clock::now();
  const nt::BpeTrainResult r = nt::train_bpe(views, opt);
  const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  std::printf("  info  nexus egitim: %.1f MB, %zu belge, %llu farkli kelime, %zu birlesim, %d is parcacigi: %.1f s\n",
              bytes / 1e6, docs.size(), static_cast<unsigned long long>(r.words), r.merges.size(), threads, secs);
  std::ifstream in(fs::path(dir) / "hf_merges.txt", std::ios::binary);
  if (in) {
    std::vector<std::string> hf;
    std::string line;
    while (std::getline(in, line)) hf.push_back(line);
    std::size_t k = 0;
    while (k < hf.size() && k < r.merges.size() && hf[k] == r.merges[k].first + " " + r.merges[k].second) ++k;
    std::printf("  info  HF ile ortak birlesim oneki: %zu / %zu (nexus %zu)\n", k, hf.size(), r.merges.size());
    nexus_test::expect(k == hf.size() && k == r.merges.size(), "gercek derlemde birlesim listesi HF ile birebir");
  }
  return 0;
}

struct MemInfo {
  double private_mb = 0.0;
  double peak_ws_mb = 0.0;
};

MemInfo mem_info() {
  MemInfo m;
#if defined(_WIN32)
  PROCESS_MEMORY_COUNTERS_EX c{};
  if (GetProcessMemoryInfo(GetCurrentProcess(), reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&c), sizeof(c))) {
    m.private_mb = c.PrivateUsage / 1048576.0;
    m.peak_ws_mb = c.PeakWorkingSetSize / 1048576.0;
  }
#endif
  return m;
}

// Manifest'teki parcalar `repeat` kez listelenir: mantiksal veri kumesi repeat kat buyur, dosyalar
// kopyalanmaz. Olculen: ozel (commit) bellek ve tepe calisma kumesi ornek sayisiyla buyuyor mu.
int bench_stream(const std::string& manifest_path, int repeat, std::uint64_t samples) {
  nexus_test::section("bench_stream: " + manifest_path + " x" + std::to_string(repeat));
  nt::Manifest m = nt::Manifest::load(manifest_path);
  const auto base = m.shards;
  for (int r = 1; r < repeat; ++r) m.shards.insert(m.shards.end(), base.begin(), base.end());
  std::uint64_t bytes = 0;
  for (const auto& s : m.shards) bytes += fs::file_size(fs::path(m.dir) / s.path);
  const MemInfo before = mem_info();
  nt::TokenStream st(m, 1024, 42);
  nt::PackedSample s;
  std::uint64_t sum = 0;
  const auto t0 = std::chrono::steady_clock::now();
  for (std::uint64_t i = 0; i < samples; ++i) {
    st.next(s);
    sum += static_cast<std::uint64_t>(s.tokens[0]);
  }
  const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  const MemInfo after = mem_info();
  std::printf("  info  veri %.2f GB (%zu parca, %.0f M token), %llu ornek T=1024: %.0f ornek/s, %.1f M token/s\n",
              bytes / 1e9, m.shards.size(), m.total_tokens() / 1e6, static_cast<unsigned long long>(samples),
              samples / secs, samples * 1024.0 / secs / 1e6);
  std::printf("  info  ozel bellek: %.1f -> %.1f MB, tepe calisma kumesi %.1f MB (kontrol %llu)\n", before.private_mb,
              after.private_mb, after.peak_ws_mb, static_cast<unsigned long long>(sum % 1000));
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  const std::string mode = argc > 1 ? argv[1] : "all";
  int rc = 0;
  try {
    if (mode == "all" || mode == "tiktoken") rc |= test_tiktoken() == 1 ? 1 : 0;
    if (mode == "all" || mode == "roundtrip") test_roundtrip();
    if (mode == "all" || mode == "train") rc |= test_train() == 1 ? 1 : 0;
    if (mode == "all" || mode == "shard") test_shard();
    if (mode == "all" || mode == "resume") rc |= test_resume(argv[0]) == 1 ? 1 : 0;
    if (mode == "resume_child") return resume_child(argc, argv);
    if (mode == "verify_dir" && argc > 2) rc |= verify_dir(argv[2]) == 1 ? 1 : 0;
    if (mode == "bench_encode" && argc > 3) return bench_encode(argv[2], argv[3]);
    if (mode == "bench_stream" && argc > 4) return bench_stream(argv[2], std::stoi(argv[3]), std::stoull(argv[4]));
    if (mode == "bench_train" && argc > 4) {
      bench_train(argv[2], static_cast<std::uint32_t>(std::stoul(argv[3])), std::stoi(argv[4]));
      return nexus_test::finish("faz8-train-bench");
    }
    if (mode == "all" || mode == "pack") test_pack();
    if (mode == "all" || mode == "ranks") test_ranks();
  } catch (const std::exception& e) {
    std::printf("  FAIL  %s\n", e.what());
    return 1;
  }
  const int f = nexus_test::finish("faz8-data");
  return f != 0 ? f : rc;
}
