"""Iki ozellik, iki sinif. C++ Trainer orneginin Python karsiligi."""

import nexus


def main() -> None:
    ozellik = 2
    ornek = 64
    egitim = []
    etiket = []
    dogrulama = []
    dogrulama_etiket = []
    for i in range(ornek):
        x = 1.0 if i % 2 == 0 else -1.0
        egitim.extend((x, 0.1 * (i % 5)))
        etiket.append(1 if x > 0.0 else 0)
        dogrulama.extend((-x, 0.05))
        dogrulama_etiket.append(1 if -x > 0.0 else 0)

    model = nexus.nn.Sequential()
    model.add(nexus.nn.Linear(2, 8, True))
    model.add(nexus.nn.swish())
    model.add(nexus.nn.Linear(8, 2, True))

    trainer = (
        nexus.train.TrainerBuilder(model)
        .adamw(model, lr=0.05)
        .loss(0.0)
        .execution_policy_cpu()
        .gradient_clip_norm(1.0)
        .accuracy()
        .early_stopping(5)
        .build()
    )
    train = nexus.data.DataLoader(egitim, etiket, ozellik, 16, True, 1)
    val = nexus.data.DataLoader(dogrulama, dogrulama_etiket, ozellik, 16, False, 1)
    trainer.fit(train, val, 12)
    trainer.save_nxs("siniflandirici.nxs")
    for tur in trainer.history():
        print(tur.epoch, tur.train_loss, tur.val_loss, tur.accuracy)


if __name__ == "__main__":
    main()
