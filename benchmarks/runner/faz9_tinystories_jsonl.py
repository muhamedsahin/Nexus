"""FAZ 9.H: TinyStoriesV2 metnini (<|endoftext|> ayracli) nexus_tokenize icin JSONL'ye cevirir.

Kullanim: python faz9_tinystories_jsonl.py GIRDI.txt CIKTI.jsonl
Her hikaye bir satir: {"text": "..."}. Bos hikayeler atlanir. Ayrac metne dahil edilmez;
nexus_tokenize --eos 50256 her belgenin sonuna <|endoftext|> tokenini ekler.

Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
olarak gelistirilmistir.
"""

import json
import sys

SEP = "<|endoftext|>"


def main() -> int:
    src, dst = sys.argv[1], sys.argv[2]
    docs = 0
    tail = ""
    with open(src, encoding="utf-8") as f, open(dst, "w", encoding="utf-8", newline="\n") as out:
        while True:
            chunk = f.read(1 << 24)
            if not chunk:
                break
            parts = (tail + chunk).split(SEP)
            tail = parts.pop()
            for p in parts:
                p = p.strip()
                if p:
                    out.write(json.dumps({"text": p}, ensure_ascii=False) + "\n")
                    docs += 1
        tail = tail.strip()
        if tail:
            out.write(json.dumps({"text": tail}, ensure_ascii=False) + "\n")
            docs += 1
    print(f"{docs} belge -> {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
