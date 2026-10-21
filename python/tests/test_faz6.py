"""Faz 6: iki Linear ve ReLU, sonlu fark 1e-4."""

import nexus


def test_teyp_sonlu_fark():
    first = nexus.nn.Linear(2, 2, True)
    second = nexus.nn.Linear(2, 1, True)
    first.parameters()[0]["values"].copy_from([0.2, 0.1, 0.0, 0.3])
    first.parameters()[1]["values"].copy_from([0.4, 0.5])
    second.parameters()[0]["values"].copy_from([0.3, -0.2])
    second.parameters()[1]["values"].copy_from([0.1])
    values = nexus.Tensor([1, 2])
    values.copy_from([0.5, -0.25])
    tape = nexus.custom.Tape()
    hidden = tape.apply(first, tape.input(values))
    output = tape.apply(second, tape.relu(hidden))
    first.zero_grad()
    second.zero_grad()
    tape.backward(output)
    analytical = first.parameters()[0]["grad"].to_list()[0]
    weight = first.parameters()[0]["values"]
    base = weight.to_list()
    eps = 1.0e-3

    def score(delta):
        changed = list(base)
        changed[0] += delta
        weight.copy_from(changed)
        hidden_value = first.forward(values)
        activated = nexus.nn.ReLU().forward(hidden_value)
        return second.forward(activated).to_list()[0]

    numeric = (score(eps) - score(-eps)) / (2.0 * eps)
    assert abs(analytical - numeric) < 1.0e-4
