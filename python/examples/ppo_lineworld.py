"""LineWorld uzerinde kisa PPO. Ikinci calistirma yalniz dosyayi yukler."""

import sys

import nexus


def main() -> None:
    path = "ppo_lineworld.nxs"
    if len(sys.argv) > 1 and sys.argv[1] == "yukle":
        ajan = nexus.rl.PPOAgent.load_nxs(path)
        eylem = ajan.act([0.0], True)
        print(eylem[0])
        return
    env = nexus.rl.LineWorld()
    config = nexus.rl.PPOConfig()
    config.hidden = 8
    config.seed = 3
    config.epochs = 1
    ajan = nexus.rl.PPOAgent(env.observation_dim(), env.action_count(), config)
    ajan.learn(env, 2, 8)
    ajan.save_nxs(path)
    print(ajan.act([0.0], True)[0])


if __name__ == "__main__":
    main()
