import unittest
from math import log
import numpy
import torch
from torch import tensor, nn
from spred.util import softmax, close_enough
from spred.confidence import inv_abstain_prob, max_nonabstain_prob
from spred.confidence import max_prob, abstention_prob, random_confidence
from spred.confidence import MCDropoutConfidence


def example_batch1():
    inputs = tensor([[0.5, -0.2, 0.8],
                     [0.2, -0.5, 0.7]])
    outputs = tensor([[-1., -2., 0., 1.],
                      [-2., 0., 1., -1.]])
    close_enough(softmax(outputs), tensor([[0.0871, 0.0321, 0.2369, 0.6439],
                                           [0.0321, 0.2369, 0.6439, 0.0871]]))
    batch = {'inputs': inputs, 'outputs': outputs}
    return batch


class ExampleModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.i = 0

    def forward(self, batch, compute_conf=True):
        batch_size = len(batch)
        outputs = tensor([[1.0 + x + self.i, 2.0 + x + self.i, 7.0 * x, 3.5 + x]
                          for x in range(batch_size)])
        loss = 100.0 - self.i
        if compute_conf:
            confidence = tensor([[0.5 + 0.01 * x for x in range(batch_size)]])
        else:
            confidence = None
        self.i += 1
        return {'outputs': outputs, 'loss': loss, 'confidences': confidence}


class TestConfidence(unittest.TestCase):

    def test_inv_abstain_prob(self):
        batch = example_batch1()
        conf = inv_abstain_prob(batch)
        expected = tensor([1.0 - .6439, 1.0 - .0871])
        close_enough(conf, expected)

    def test_max_nonabstain_prob(self):
        batch = example_batch1()
        conf = max_nonabstain_prob(batch)
        expected = tensor([0.2369, 0.6439])
        close_enough(conf, expected)

    def test_max_prob(self):
        batch = example_batch1()
        conf = max_prob(batch)
        expected = tensor([0.6439, 0.6439])
        close_enough(conf, expected)

    def test_abstention_prob(self):
        batch = example_batch1()
        conf = abstention_prob(batch)
        expected = tensor([0.6439, 0.0871])
        close_enough(conf, expected)

    def test_random_confidence(self):
        batch = example_batch1()
        conf = random_confidence(batch)
        assert conf.shape == (2,)
        assert 0.0 <= conf[0] <= 1.0
        assert 0.0 <= conf[1] <= 1.0

    def test_mc_dropout_mean(self):
        base_model = ExampleModel()
        conf_fn = MCDropoutConfidence(base_model, n_forward_passes=4,
                                      combo_fn=torch.mean)
        batch = example_batch1()
        expected = tensor([(0.7488 + 0.5377 + 0.3044 + 0.1397) / 4.0,
                           (0.9032 + 0.8694 + 0.7891 + 0.6308) / 4.0])
        close_enough(conf_fn(batch), expected)

    def test_mc_dropout_variance(self):
        base_model = ExampleModel()
        conf_fn = MCDropoutConfidence(base_model, n_forward_passes=4,
                                      combo_fn=torch.var)
        batch = example_batch1()
        expected = tensor([numpy.var([0.7488, 0.5377, 0.3044, 0.1397], ddof=1),
                           numpy.var([0.9032, 0.8694, 0.7891, 0.6308], ddof=1)]).float()
        close_enough(conf_fn(batch), expected)


if __name__ == "__main__":
    unittest.main()
