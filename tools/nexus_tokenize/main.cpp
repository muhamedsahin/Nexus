/**
 * @file main.cpp
 * @brief FAZ 8.2 nexus_tokenize: ham metin/JSONL -> NXTK parcalari + manifest.
 *
 * Kullanim:
 *   nexus_tokenize --vocab gpt2.tiktoken --pretokenizer gpt2 --eos 50256 --out DIR
 *                  [--shard-tokens 100000000] [--threads 0] [--format auto|txt|jsonl]
 *                  [--text-field text] [--split-blank] [--dedup] --input YOL [YOL ...]
 * YOL bir dosya ya da dosya adinda * / ? iceren bir desen (dizin kismi joker icermez).
 * txt: dosya basina bir belge (--split-blank: bos satirla ayrilmis belgeler). jsonl: satir basina
 * bir nesne, metin `--text-field` alaninda. --dedup: tam belge SHA-256'siyla tekrarlar atilir.
 * Parquet / WebDataset / uzak URL girdileri ve MinHash yakin-kopya bu surumde yok.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexusdata/io/json.hpp"
#include "nexusdata/text/bpe.hpp"
#include "nexusdata/text/token_shard.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace nt = nexusdata::text;
namespace fs = std::filesystem;

namespace {

struct Options {
  std::string vocab, pretokenizer = "gpt2", out, format = "auto", text_field = "text";
  std::int64_t eos = -1;
  std::uint64_t shard_tokens = 100'000'000;
  int threads = 0;
  bool split_blank = false;
  bool dedup = false;
  std::vector<std::string> inputs;
};

[[noreturn]] void usage(const char* why) {
  std::fprintf(stderr, "nexus_tokenize: %s\n(bkz. tools/nexus_tokenize/main.cpp basligi)\n", why);
  std::exit(2);
}

std::vector<fs::path> expand(const std::string& pattern) {
  const fs::path p(pattern);
  const std::string name = p.filename().string();
  if (name.find_first_of("*?") == std::string::npos) {
    if (!fs::exists(p)) usage(("girdi yok: " + pattern).c_str());
    return {p};
  }
  std::string re;
  for (const char c : name) {
    if (c == '*') re += ".*";
    else if (c == '?') re += '.';
    else if (std::strchr(".+()[]{}^$|\\", c) != nullptr) { re += '\\'; re += c; }
    else re += c;
  }
  const std::regex rx(re);
  std::vector<fs::path> out;
  const fs::path dir = p.has_parent_path() ? p.parent_path() : fs::path(".");
  for (const auto& e : fs::directory_iterator(dir)) {
    if (e.is_regular_file() && std::regex_match(e.path().filename().string(), rx)) out.push_back(e.path());
  }
  std::sort(out.begin(), out.end());
  return out;
}

std::string read_all(const fs::path& p) {
  std::ifstream in(p, std::ios::binary);
  std::stringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

}  // namespace

int main(int argc, char** argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto val = [&]() -> std::string {
      if (i + 1 >= argc) usage(("deger eksik: " + a).c_str());
      return argv[++i];
    };
    if (a == "--vocab") o.vocab = val();
    else if (a == "--pretokenizer") o.pretokenizer = val();
    else if (a == "--eos") o.eos = std::stoll(val());
    else if (a == "--out") o.out = val();
    else if (a == "--shard-tokens") o.shard_tokens = std::stoull(val());
    else if (a == "--threads") o.threads = std::stoi(val());
    else if (a == "--format") o.format = val();
    else if (a == "--text-field") o.text_field = val();
    else if (a == "--split-blank") o.split_blank = true;
    else if (a == "--dedup") o.dedup = true;
    else if (a == "--input") {
      while (i + 1 < argc && std::strncmp(argv[i + 1], "--", 2) != 0) o.inputs.emplace_back(argv[++i]);
    } else usage(("bilinmeyen secenek: " + a).c_str());
  }
  if (o.vocab.empty() || o.out.empty() || o.inputs.empty()) usage("--vocab, --out ve --input zorunlu");
  try {
    const nt::ByteBpe bpe =
        nt::ByteBpe::load_tiktoken(o.vocab, nt::pretokenizer_from_string(o.pretokenizer),
                                   o.eos >= 0 ? std::vector<std::pair<std::string, std::uint32_t>>{{"<eos>", static_cast<std::uint32_t>(o.eos)}}
                                              : std::vector<std::pair<std::string, std::uint32_t>>{});
    fs::create_directories(o.out);
    const auto digest = bpe.digest();
    nt::Manifest manifest;
    manifest.tokenizer_name = o.pretokenizer;
    manifest.tokenizer_digest = nt::to_hex(digest);
    manifest.vocab_size = bpe.vocab_size();
    manifest.eos = o.eos;
    std::unique_ptr<nt::ShardWriter> writer;
    std::string shard_name;
    auto close_shard = [&] {
      if (!writer) return;
      const auto sha = writer->finish();
      manifest.shards.push_back({shard_name, writer->tokens(), writer->docs(), nt::to_hex(sha)});
      writer.reset();
    };
    std::set<std::string> seen;
    std::uint64_t docs = 0, dups = 0, bytes = 0;
    std::vector<std::string> batch;
    auto flush = [&] {
      std::vector<std::string_view> views(batch.begin(), batch.end());
      const auto encoded = bpe.encode_batch(views, o.threads);
      for (const auto& ids : encoded) {
        if (!writer) {
          char name[32];
          std::snprintf(name, sizeof(name), "shard_%05zu.nxtk", manifest.shards.size());
          shard_name = name;
          writer = std::make_unique<nt::ShardWriter>((fs::path(o.out) / shard_name).string(), bpe.vocab_size(), digest, o.eos);
        }
        writer->add_document(ids.data(), ids.size());
        if (writer->tokens() >= o.shard_tokens) close_shard();
      }
      batch.clear();
    };
    auto add_doc = [&](std::string text) {
      if (o.dedup) {
        const std::string key = nt::to_hex([&] {
          nt::Sha256 h;
          h.update(text.data(), text.size());
          return h.finish();
        }());
        if (!seen.insert(key).second) {
          ++dups;
          return;
        }
      }
      bytes += text.size();
      ++docs;
      batch.push_back(std::move(text));
      if (batch.size() >= 4096) flush();
    };
    const auto t0 = std::chrono::steady_clock::now();
    for (const std::string& pat : o.inputs) {
      for (const fs::path& f : expand(pat)) {
        const bool jsonl = o.format == "jsonl" || (o.format == "auto" && f.extension() == ".jsonl");
        if (jsonl) {
          std::ifstream in(f, std::ios::binary);
          std::string line;
          while (std::getline(in, line)) {
            if (line.empty()) continue;
            const auto v = nexusdata::json::parse(line);
            const auto* t = v.find(o.text_field);
            if (t != nullptr && t->is_string()) add_doc(t->as_string());
          }
        } else {
          std::string text = read_all(f);
          if (!o.split_blank) {
            add_doc(std::move(text));
            continue;
          }
          std::size_t a = 0;
          while (a < text.size()) {
            std::size_t b = text.find("\n\n", a);
            if (b == std::string::npos) b = text.size();
            if (b > a) add_doc(text.substr(a, b - a));
            a = b + 2;
          }
        }
      }
    }
    flush();
    close_shard();
    manifest.save((fs::path(o.out) / "manifest.json").string());
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("nexus_tokenize: %llu belge (%llu tekrar atildi), %.1f MB, %llu token, %zu parca, %.1f s (%.1f MB/s)\n",
                static_cast<unsigned long long>(docs), static_cast<unsigned long long>(dups), bytes / 1e6,
                static_cast<unsigned long long>(manifest.total_tokens()), manifest.shards.size(), secs, bytes / 1e6 / secs);
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "nexus_tokenize: %s\n", e.what());
    return 1;
  }
}
