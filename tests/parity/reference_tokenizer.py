"""FAZ 8 tokenizer referansi: tiktoken (kodlama) ve HF tokenizers (BPE egitimi).

Kullanim:
  python reference_tokenizer.py export <dir>          # gpt2.tiktoken, cl100k.tiktoken
  python reference_tokenizer.py encode <dir> <name>   # docs.bin -> ref_<name>.bin
  python reference_tokenizer.py train <dir> <vocab>   # corpus.bin -> hf_merges.txt, hf_vocab.txt
  python reference_tokenizer.py bench <dir> <name>    # docs.bin, tiktoken 1/8/16 is parcacigi MB/s
  python reference_tokenizer.py benchhf <dir> <th>    # docs.bin, HF tokenizers GPT-2, RAYON_NUM_THREADS=th
  python reference_tokenizer.py corpus <dir> <MB> <kok...>  # yerel dosyalardan docs.bin

docs.bin: uint32 belge sayisi, sonra her belge icin uint32 bayt uzunlugu + UTF-8 baytlari.
ref_<name>.bin: her belge icin uint32 token sayisi + uint32 token'lar.
"""
import base64
import json
import os
import struct
import sys
import time


def read_docs(path):
    with open(path, "rb") as f:
        data = f.read()
    (n,) = struct.unpack_from("<I", data, 0)
    off = 4
    docs = []
    for _ in range(n):
        (ln,) = struct.unpack_from("<I", data, off)
        off += 4
        docs.append(data[off:off + ln].decode("utf-8"))
        off += ln
    return docs


def export(d):
    import tiktoken
    for name, enc in (("gpt2", "gpt2"), ("cl100k", "cl100k_base")):
        e = tiktoken.get_encoding(enc)
        with open(os.path.join(d, f"{name}.tiktoken"), "w", encoding="ascii") as f:
            for tok, rank in sorted(e._mergeable_ranks.items(), key=lambda kv: kv[1]):
                f.write(base64.b64encode(tok).decode("ascii") + f" {rank}\n")
        with open(os.path.join(d, f"{name}.specials.json"), "w", encoding="utf-8") as f:
            json.dump(e._special_tokens, f)


def encode(d, name):
    import tiktoken
    e = tiktoken.get_encoding({"gpt2": "gpt2", "cl100k": "cl100k_base"}[name])
    docs = read_docs(os.path.join(d, "docs.bin"))
    out = bytearray()
    for doc in docs:
        ids = e.encode_ordinary(doc)
        out += struct.pack("<I", len(ids))
        out += struct.pack(f"<{len(ids)}I", *ids)
    with open(os.path.join(d, f"ref_{name}.bin"), "wb") as f:
        f.write(out)


def train(d, vocab):
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    docs = read_docs(os.path.join(d, "corpus.bin"))
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    trainer = trainers.BpeTrainer(vocab_size=int(vocab), min_frequency=0, show_progress=False,
                                  special_tokens=["<|endoftext|>"],
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    t0 = time.perf_counter()
    tok.train_from_iterator(docs, trainer=trainer)
    secs = time.perf_counter() - t0
    model = json.loads(tok.to_str())["model"]
    with open(os.path.join(d, "hf_merges.txt"), "w", encoding="utf-8", newline="\n") as f:
        for m in model["merges"]:
            a, b = (m.split(" ", 1) if isinstance(m, str) else m)
            f.write(f"{a} {b}\n")
    with open(os.path.join(d, "hf_vocab.txt"), "w", encoding="utf-8", newline="\n") as f:
        for t, i in sorted(model["vocab"].items(), key=lambda kv: kv[1]):
            f.write(f"{i} {t}\n")
    with open(os.path.join(d, "hf_train_time.txt"), "w") as f:
        f.write(f"{secs}\n")


def corpus(d, max_mb, roots):
    """Yerel kaynak/dokuman dosyalarindan docs.bin (dosya basina bir belge, sirali, belirlenimci)."""
    exts = {".py", ".pyi", ".go", ".c", ".h", ".cc", ".cpp", ".hpp", ".cu", ".cuh", ".java", ".js", ".ts",
            ".rs", ".md", ".rst", ".txt", ".html", ".htm", ".xml", ".json", ".css", ".cmake", ".inl", ".s"}
    os.makedirs(d, exist_ok=True)
    limit = int(float(max_mb) * 1e6)
    total = 0
    info = {}
    out = open(os.path.join(d, "docs.bin"), "wb")
    out.write(struct.pack("<I", 0))
    count = 0
    for root in roots:
        got = 0
        for dirpath, dirnames, files in os.walk(root):
            dirnames.sort()
            for fn in sorted(files):
                if os.path.splitext(fn)[1].lower() not in exts:
                    continue
                try:
                    if os.path.getsize(os.path.join(dirpath, fn)) > 2_000_000:
                        continue
                    with open(os.path.join(dirpath, fn), "rb") as f:
                        raw = f.read()
                    raw.decode("utf-8")
                except (OSError, UnicodeDecodeError):
                    continue
                if not raw or total + len(raw) > limit:
                    continue
                out.write(struct.pack("<I", len(raw)))
                out.write(raw)
                total += len(raw)
                got += len(raw)
                count += 1
            if total >= limit:
                break
        info[root] = got
    out.seek(0)
    out.write(struct.pack("<I", count))
    out.close()
    with open(os.path.join(d, "corpus_info.json"), "w") as f:
        json.dump({"bytes": total, "docs": count, "roots": info}, f, indent=1)
    print(f"corpus: {count} belge, {total / 1e6:.1f} MB")


def bench(d, name):
    import tiktoken
    e = tiktoken.get_encoding({"gpt2": "gpt2", "cl100k": "cl100k_base"}[name])
    docs = read_docs(os.path.join(d, "docs.bin"))
    nbytes = sum(len(x.encode("utf-8")) for x in docs)
    res = {"bytes": nbytes, "docs": len(docs), "tiktoken": {}}
    for th in (1, 8, 16):
        e.encode_ordinary_batch(docs[:200], num_threads=th)  # isinma
        best = 1e9
        for _ in range(3):
            t0 = time.perf_counter()
            e.encode_ordinary_batch(docs, num_threads=th)
            best = min(best, time.perf_counter() - t0)
        res["tiktoken"][str(th)] = nbytes / best / 1e6
    with open(os.path.join(d, f"bench_{name}_tiktoken.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res))


def bench_hf(d, threads):
    """HF tokenizers GPT-2 (tiktoken sozlugunden kurulur). RAYON_NUM_THREADS ice aktarmadan once."""
    os.environ["RAYON_NUM_THREADS"] = str(threads)
    import tiktoken
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    e = tiktoken.get_encoding("gpt2")
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(0xA1, 0xAD)) + list(range(0xAE, 0x100))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    b2u = {b: chr(c) for b, c in zip(bs, cs)}
    u = lambda t: "".join(b2u[x] for x in t)
    ranks = e._mergeable_ranks
    vocab = {u(t): r for t, r in ranks.items()}
    # HF TikTokenConverter yontemi: her token icin TUM gecerli bolunmeler (iki parca da sozlukte),
    # birlesik token rank'i, sonra sol/sag parca rank'ina gore sirali.
    merges = []
    for t, r in sorted(ranks.items(), key=lambda kv: kv[1]):
        if len(t) < 2:
            continue
        local = sorted((ranks[t[:k]], ranks[t[k:]], k) for k in range(1, len(t)) if t[:k] in ranks and t[k:] in ranks)
        merges += [(u(t[:k]), u(t[k:])) for _, _, k in local]
    tok = Tokenizer(models.BPE(vocab=vocab, merges=merges))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tok.decoder = decoders.ByteLevel()
    docs = read_docs(os.path.join(d, "docs.bin"))
    nbytes = sum(len(x.encode("utf-8")) for x in docs)
    sample = docs[:2000]
    mismatch = sum(1 for a, b in zip(tok.encode_batch(sample, add_special_tokens=False), e.encode_ordinary_batch(sample))
                   if a.ids != b)
    best = 1e9
    for _ in range(1 if threads == 1 else 3):
        t0 = time.perf_counter()
        tok.encode_batch(docs, add_special_tokens=False)
        best = min(best, time.perf_counter() - t0)
    res = {"threads": threads, "mbps": nbytes / best / 1e6, "mismatch_docs_of_2000": mismatch}
    with open(os.path.join(d, f"bench_gpt2_hf_{threads}.json"), "w") as f:
        json.dump(res, f)
    print(json.dumps(res))


if __name__ == "__main__":
    mode, d = sys.argv[1], sys.argv[2]
    {"export": lambda: export(d), "encode": lambda: encode(d, sys.argv[3]),
     "train": lambda: train(d, sys.argv[3]), "bench": lambda: bench(d, sys.argv[3]),
     "benchhf": lambda: bench_hf(d, int(sys.argv[3])),
     "corpus": lambda: corpus(d, sys.argv[3], sys.argv[4:])}[mode]()
