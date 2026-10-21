# Faz 5 pekistirmeli ogrenme

Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli olarak gelistirilmistir.

Basliklar `include/nexus/rl/` altindadir.

## Ortam

`Environment::reset` ve `step` bir `StepResult` dondurur: gozlem, odul, terminated, truncated. `Discrete`, `Box` ve `MultiDiscrete` uzaylari ornegin bu araliga dustugunu denetler. `VectorEnv` kopyalari ayri nesnelerde kosar, gozlemleri `[kopya, boyut]` tensörunde istifler ve biten bolumu yeniden baslatir. Sekiz ve daha cok kopyada adimlar kalici iscilere dagitilir; her adim yeni thread acmaz.

`MultiAgentEnvironment::step` butun ajanlarin eylemini ayni anda uygular. `TurnLine` AEC'dir: `current_agent` ve `step_one` sirayla ilerler.

Yerlesik ortamlar: `LineWorld`, `GridWorld`, `CartPole`, `PointMass`, isbirlikci `ButtonWorld`, sifir toplam `Duel`.

## Tamponlar

`ReplayBuffer` 64 bayt hizali bir halkadir. `PrioritizedReplayBuffer` toplam agaciyla O(log N) ornekler ve importance sampling agirligi uretir. `RolloutBuffer::finish` GAE ve avantaj normalizasyonu yapar. Terminal adim bootstrap etmez.

## Ajanlar

`DQNAgent` Double DQN, Dueling baslik ve PER seceneklerini tasir. `PPOAgent` kirpilmis vekil amac, deger kirpmasi ve entropi bonusu kullanir. `ActorCriticAgent::discrete` kirpmayi kapatip tek epok kosar. `ContinuousActorCritic` ve `SACAgent` `PointMass` uzerinde surekli eylemi ogrenir.

`MultiAgentTrainer::train_independent` IPPO, `train_centralized` MAPPO'dur. Ajan sayisi ortamin `agents()` listesinden gelir; gozlem boyu ve eylem sayisi ortak olmalidir. Merkezi kritik gozlemleri art arda birlestirir; aktorler yalnizca kendi gozlemlerini kullanir. `TeamButton` N ajanin ayni anda basmasini ister. `League` ELO ve agirlik anlik goruntusu tutar.

128 LineWorld kopyasi, 200 adim: 549 119 adim/s (1.82 µs/adim). DQN batch 128: 1868 guncelleme/s.
