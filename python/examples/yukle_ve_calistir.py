"""Kaydedilmis .nxs dosyasini kurup bir ileri gecis yapar."""

import sys

import nexus


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "siniflandirici.nxs"
    model = nexus.load(path)
    girdi = nexus.Tensor([1, 2])
    girdi[0] = 1.0
    girdi[1] = 0.1
    cikti = model.forward(girdi)
    print(cikti.to_list())


if __name__ == "__main__":
    main()
