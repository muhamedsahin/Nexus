"""Cok girisli graf ve kaydirilmis token kaybi."""

import nexus


def _identity(layer, width):
    weight = [0.0] * (width * width)
    for index in range(width):
        weight[index * width + index] = 1.0
    layer.parameters()[0]["values"].copy_from(weight)
    if len(layer.parameters()) > 1:
        bias = layer.parameters()[1]["values"]
        bias.copy_from([0.0] * bias.numel())


def test_graph_reads_two_nodes():
    left = nexus.nn.Linear(2, 2, True)
    right = nexus.nn.Linear(2, 2, True)
    joined = nexus.nn.Linear(4, 4, True)
    _identity(left, 2)
    _identity(right, 2)
    _identity(joined, 4)
    graph = nexus.nn.Graph()
    graph.add("katman2", left)
    graph.add("katman3", nexus.nn.ReLU())
    graph.add("katman4", right)
    graph.concat("katman5", joined, ["katman4", "katman2"])
    values = nexus.Tensor([1, 2])
    values[0] = 3.0
    values[1] = 5.0
    logits = graph.forward(values).to_list()
    assert len(logits) == 4
    for got, want in zip(logits, (3.0, 5.0, 3.0, 5.0)):
        assert abs(got - want) < 1.0e-5


def test_language_loss_drops():
    vocab = 6
    embed = 8
    model = nexus.nn.Sequential()
    model.add(nexus.nn.Embedding(vocab, embed, -1))
    model.add(nexus.nn.PositionalEncoding(embed, 32, False))
    model.add(nexus.nn.CausalTransformerBlock(embed, 2, 16))
    model.add(nexus.nn.Linear(embed, vocab, True))
    trainer = nexus.train.LanguageTrainer(model, 0.08)
    tokens = [[time % vocab for time in range(8)] for _ in range(4)]
    first = trainer.fit(tokens, 1)
    later = trainer.fit(tokens, 30)
    assert later < first


if __name__ == "__main__":
    test_graph_reads_two_nodes()
    test_language_loss_drops()
