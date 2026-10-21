"""Faz 1: baglanmamis C++ yuzeyi."""

import nexus


def test_parametre_ayni_depo():
    model = nexus.nn.Sequential()
    model.add(nexus.nn.Linear(2, 3, True))
    params = model.parameters()
    assert len(params) == 2
    values = params[0]["values"]
    before = values.to_list()
    values.copy_from([1.0] * len(before))
    assert model.parameters()[0]["values"].to_list() == [1.0] * len(before)
    model.zero_grad()


def test_sgd_hiz_ve_kayip():
    model = nexus.nn.Sequential()
    model.add(nexus.nn.Linear(2, 2, True))
    opt = nexus.optim.SGD(model, lr=0.1, momentum=0.0, weight_decay=0.0)
    assert abs(opt.learning_rate() - 0.1) < 1.0e-6
    opt.set_learning_rate(0.02)
    assert abs(opt.learning_rate() - 0.02) < 1.0e-6
    opt.zero_grad()
    loss = nexus.loss.CrossEntropy(0.0).forward_backward([2.0, 0.0, 0.0, 2.0], [0, 1], 2)
    assert loss["loss"] > 0.0
    assert len(loss["grad"]) == 4
    metric = nexus.metrics.Accuracy()
    metric.update([2.0, 0.0, 0.0, 2.0], [0, 1], 2)
    assert abs(metric.value() - 1.0) < 1.0e-6
    metric.reset()
    assert metric.value() == 0.0


def test_yukleyici_ve_zamanlayici():
    loader = nexus.data.DataLoader([1.0, 0.0, 0.0, 1.0], [0, 1], 2, 1, False, 1)
    assert loader.features_per_sample() == 2
    first = loader.next()
    assert first["batch_size"] == 1
    assert loader.next()["labels"] == [1]
    assert loader.next() is None
    loader.reset()
    assert loader.next()["labels"] == [0]
    model = nexus.nn.Sequential()
    model.add(nexus.nn.Linear(2, 2, True))
    schedule = nexus.optim.CosineAnnealing(0.1, 2)
    logger = nexus.train.ProgressLogger()
    trainer = (
        nexus.train.TrainerBuilder(model)
        .adamw(model, lr=0.1)
        .loss()
        .cosine(schedule)
        .callback(logger)
        .mixed_precision(False)
        .build()
    )
    trainer.fit(loader, loader, 1)
    assert logger.last()["epoch"] == 0


def test_tampon_ve_eylem():
    buffer = nexus.rl.ReplayBuffer(4, 2)
    buffer.add([0.0, 1.0], 1, 0.5, [1.0, 0.0], False)
    sample = buffer.sample([0])
    assert sample["actions"] == [1]
    assert abs(sample["rewards"][0] - 0.5) < 1.0e-6
    per = nexus.rl.PrioritizedReplayBuffer(4, 2, 0.6)
    per.add([0.0, 1.0], 0, 1.0, [1.0, 0.0], False, 1.0)
    drawn = per.sample(1, 0.4, 3)
    assert drawn["actions"] == [0]
    rollout = nexus.rl.RolloutBuffer()
    rollout.reset(1, 2)
    rollout.add([0.2, 0.3], 1, -0.1, 0.4, 1.0, False)
    rollout.finish(0.99, 0.95, 0.0)
    assert rollout.size() == 1
    assert rollout.actions()[0] == 1
    agent = nexus.rl.ActorCriticAgent.discrete(2, 2, 1)
    action, log_prob, value = agent.act([0.0, 1.0], True)
    assert action in (0, 1)
    assert isinstance(log_prob, float) and isinstance(value, float)
    continuous = nexus.rl.ContinuousActorCritic(1, 1)
    act = continuous.act([0.25], True)
    assert -1.0 <= act <= 1.0
    tape = nexus.custom.MicroTape()
    tape.input(1.0)
    assert tape.size() == 1
    tape.reset()
    assert tape.size() == 0
