import unittest
from math import log
import numpy
import torch
from torch import tensor, nn
from spred.util import softmax, close_enough, approx
from spred.confidence import inv_abstain_prob, max_nonabstain_prob
from spred.confidence import max_prob, random_confidence
from spred.confidence import MCDropoutConfidence, TrustScore
from spred.confidence import k_nearest_neighbors, high_density_set
from spred.confidence import group_by_label, compute_high_density_sets
from spred.confidence import distance_to_set


def example_batch1():
    inputs = tensor([[0.5, -0.2, 0.8],
                     [0.2, -0.5, 0.7]])
    outputs = tensor([[-1., -2., 0., 1.],
                      [-2., 0., 1., -1.]])
    close_enough(softmax(outputs), tensor([[0.0871, 0.0321, 0.2369, 0.6439],
                                           [0.0321, 0.2369, 0.6439, 0.0871]]))
    batch = {'inputs': inputs, 'outputs': outputs}
    return batch

def example_batch2():
    inputs = tensor([[5.0, 2.0],
                     [-4.0, -2.0]])
    outputs = tensor([[-1., -2., 0.],
                      [3., 0., 1.]])
    close_enough(softmax(outputs), tensor([[0.2447, 0.0900, 0.6652],
                                           [0.8438, 0.0420, 0.1142]]))

    batch = {'inputs': inputs, 'outputs': outputs}
    return batch

class ExampleModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.i = 0

    def forward(self, batch, compute_conf=True, compute_loss=True):
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

    def test_random_confidence(self):
        batch = example_batch1()
        conf = random_confidence(batch)
        assert conf.shape == (2,)
        assert 0.0 <= conf[0] <= 1.0
        assert 0.0 <= conf[1] <= 1.0

    def test_mc_dropout_mean(self):
        base_model = ExampleModel()
        conf_fn = MCDropoutConfidence(n_forward_passes=4,
                                      combo_fn=torch.mean)
        batch = example_batch1()
        expected = tensor([(0.7488 + 0.5377 + 0.3044 + 0.1397) / 4.0,
                           (0.9032 + 0.8694 + 0.7891 + 0.6308) / 4.0])
        close_enough(conf_fn(batch, base_model), expected)

    def test_mc_dropout_variance(self):
        base_model = ExampleModel()
        conf_fn = MCDropoutConfidence(n_forward_passes=4,
                                      combo_fn=torch.var)
        batch = example_batch1()
        expected = tensor([numpy.var([0.7488, 0.5377, 0.3044, 0.1397], ddof=1),
                           numpy.var([0.9032, 0.8694, 0.7891, 0.6308], ddof=1)]).float()
        close_enough(conf_fn(batch, base_model), expected)

    def test_nearest_neighbors(self):
        t = tensor([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
        expected = tensor([[0, 1], [1, 0], [2, 1], [3, 4], [4, 3], [5, 4]])
        close_enough(k_nearest_neighbors(t, 2), expected)
        expected = tensor([[0, 1, 2],
                           [1, 0, 2],
                           [2, 1, 0],
                           [3, 4, 5],
                           [4, 3, 5],
                           [5, 4, 3]])
        close_enough(k_nearest_neighbors(t, 3), expected)

    def test_high_density_set(self):
        t = tensor([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
        expected = tensor([[-1, -1], [-2, -1], [1, 1], [2, 1]])
        close_enough(high_density_set(t, 2, 0.67), expected)

    def test_group_by_label(self):
        batch = {'inputs': tensor([[ 0.1019, -0.9391],
                                   [-0.9615, -0.2013],
                                   [ 0.3892, -0.1612],
                                   [-0.1101,  0.1560],
                                   [-0.2478, -0.4136],
                                   [-0.6994,  1.2889],
                                   [ 0.9901, -0.1670],
                                   [ 0.4204, -1.8723]]),
                 'labels': tensor([1, 0, 2, 1, 0, 2, 0, 1])}
        expected = {0: tensor([[-0.9615, -0.2013],
                               [-0.2478, -0.4136],
                               [ 0.9901, -0.1670]]),
                    1: tensor([[ 0.1019, -0.9391],
                               [-0.1101,  0.1560],
                               [ 0.4204, -1.8723]]),
                    2: tensor([[ 0.3892, -0.1612],
                               [-0.6994,  1.2889]])}
        grouped = group_by_label(batch)
        assert grouped.keys() == expected.keys()
        close_enough(grouped[0], expected[0])
        close_enough(grouped[1], expected[1])
        close_enough(grouped[2], expected[2])

    def test_compute_high_density_sets(self):
        t = tensor([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2],
                    [-3, -1], [-4, -1], [-5, -2], [-1, 1], [0, 1], [1, 2]])
        batch = {'inputs': t,
                 'labels': tensor([1, 1, 1, 1, 1, 1,
                                   0, 0, 0, 0, 0, 0])}
        hd_sets = compute_high_density_sets(batch, 2, 0.67)
        assert set(hd_sets.keys()) == {0,1}
        close_enough(hd_sets[0], tensor([[-3, -1], [-4, -1], [-1, 1], [0, 1]]))
        close_enough(hd_sets[1], tensor([[-1, -1], [-2, -1], [1, 1], [2, 1]]))

    def test_distance_to_set(self):
        t = tensor([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
        assert approx(distance_to_set(tensor([-1, -1]), t), 0.0)
        assert approx(distance_to_set(tensor([3, 2]), t), 0.0)
        assert approx(distance_to_set(tensor([4, 2]), t), 1.0)
        assert approx(distance_to_set(tensor([5, 2]), t), 2.0)

    def test_trustscore(self):
        t = tensor([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2],
                    [-3, -1], [-4, -1], [-5, -2], [-1, 1], [0, 1], [1, 2],
                    [50, 20], [52, 20], [50, 21], [53, 20], [50, 25]])
        train_batch = {'inputs': t,
                       'labels': tensor([2, 2, 2, 2, 2, 2,
                                         0, 0, 0, 0, 0, 0,
                                         1, 1, 1, 1, 1])}
        batch = example_batch2()
        score = TrustScore(train_batch, 2, 1.0)
        close_enough(score(batch), tensor([2., 1.]).double())
        score = TrustScore(train_batch, 2, 0.67)
        close_enough(score(batch), tensor([1.6125, 2.2361]).double())


if __name__ == "__main__":
    unittest.main()
