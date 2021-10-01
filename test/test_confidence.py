import unittest
from math import log
import numpy
import torch
from torch import tensor, nn
from spred.loader import Loader
from spred.util import softmax, close_enough, approx
from spred.confidence import sum_nonabstain_prob, max_nonabstain_prob
from spred.confidence import max_prob, random_confidence
from spred.confidence import MCDropoutConfidence, TrustScore
from test.examples import ExampleLoader


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

    def test_sum_nonabstain_prob(self):
        batch = example_batch1()
        conf = sum_nonabstain_prob(batch)
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
                                      combo_id="mean")
        batch = example_batch1()
        expected = tensor([(0.7488 + 0.5377 + 0.3044 + 0.1397) / 4.0,
                           (0.9032 + 0.8694 + 0.7891 + 0.6308) / 4.0])
        close_enough(conf_fn(batch, base_model), expected)

    def test_mc_dropout_variance(self):
        base_model = ExampleModel()
        conf_fn = MCDropoutConfidence(n_forward_passes=4,
                                      combo_id="negvar")
        batch = example_batch1()
        expected = tensor([-1 * numpy.var([0.7488, 0.5377, 0.3044, 0.1397], ddof=1),
                           -1 * numpy.var([0.9032, 0.8694, 0.7891, 0.6308], ddof=1)]).float()
        close_enough(conf_fn(batch, base_model), expected)

    def test_high_density_set(self):
        t = tensor([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
        expected = tensor([[-1, -1], [-2, -1], [1, 1], [2, 1]])
        close_enough(TrustScore.high_density_set(t, 2, 0.33), expected)

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
        grouped = TrustScore.group_by_label(batch)
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
        hd_sets = TrustScore.compute_high_density_sets(batch, 2, 0.33)
        assert set(hd_sets.keys()) == {0,1}
        close_enough(hd_sets[0], tensor([[-3, -1], [-4, -1], [-1, 1], [0, 1]]))
        close_enough(hd_sets[1], tensor([[-1, -1], [-2, -1], [1, 1], [2, 1]]))

    def test_distance_to_set(self):
        t = tensor([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
        assert approx(TrustScore.distance_to_set(tensor([-1, -1]), t), 0.0)
        assert approx(TrustScore.distance_to_set(tensor([3, 2]), t), 0.0)
        assert approx(TrustScore.distance_to_set(tensor([4, 2]), t), 1.0)
        assert approx(TrustScore.distance_to_set(tensor([5, 2]), t), 2.0)

    def test_trustscore1(self):
        class TrivialEmbedder:
            def embed(self, x):
                return {'outputs': x['inputs']}

            def eval(self):
                pass

        batch1 = {'inputs': tensor([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [5, 6]]),
                  'labels': tensor([1, 1, 1, 1, 1, 1])}
        batch2 = {'inputs': tensor([[-3, -1], [-4, -1], [-5, -2], [-1, 1], [0, 1], [7, 6]]),
                  'labels': tensor([0, 0, 0, 0, 0, 0])}
        model_out = {'inputs': tensor([[6.0, 6.0],
                                       [8.0, 6.0],
                                       [8.0, 6.0]]),
                     'outputs': tensor([[-1., -2.],
                                        [3., 0.],
                                        [0., 5.]])}
        train_loader = ExampleLoader([batch1, batch2], output_size=2)
        score = TrustScore(train_loader, TrivialEmbedder(), 2, 0.0)
        close_enough(score(model_out), tensor([1., 3., .3333]).double())
        train_loader = ExampleLoader([batch1, batch2], output_size=2)
        score = TrustScore(train_loader, TrivialEmbedder(), 2, 0.33)
        close_enough(score(model_out), tensor([0.8198, 0.8279, 1.2079]).double())



if __name__ == "__main__":
    unittest.main()
