import unittest
from math import log
import torch
from torch import tensor
from torch.nn import functional
from spred.util import abstain_probs, nonabstain_probs
from spred.util import renormalized_nonabstain_probs
from spred.util import nonabstain_prob_mass, gold_values


def approx(x, y, num_digits=4):
    return abs(x-y) < 1.0 * (10 ** -num_digits)


def softmax(t):
    return functional.softmax(t.clamp(min=-25, max=25), dim=1)


def close_enough(t1, t2):
    error_msg = "Not close enough:\n{}\n{}".format(t1, t2)
    assert torch.allclose(t1, t2, atol=0.001), error_msg


class TestUtil(unittest.TestCase):

    """
    For the following tests, if we apply softmax to:
        tensor([[-1., -2., 0., 1.],
                [-2., 0., 1., -1.]]
    we obtain:
        tensor([[0.0871, 0.0321, 0.2369, 0.6439],
                [0.0321, 0.2369, 0.6439, 0.0871]])
    """
    def test_abstain_probs(self):
        t = tensor([[-1., -2., 0., 1.],
                    [-2., 0., 1., -1.]])
        expected = tensor([0.6439, 0.0871])
        close_enough(abstain_probs(t), expected)

    def test_nonabstain_probs(self):
        t = tensor([[-1., -2., 0., 1.],
                    [-2., 0., 1., -1.]])
        expected = tensor([[0.0871, 0.0321, 0.2369],
                           [0.0321, 0.2369, 0.6439]])
        close_enough(nonabstain_probs(t), expected)

    def test_nonabstain_prob_mass(self):
        t = tensor([[-1., -2., 0., 1.],
                    [-2., 0., 1., -1.]])
        expected = tensor([0.0871 + 0.0321 + 0.2369,
                           0.0321 + 0.2369 + 0.6439])
        close_enough(nonabstain_prob_mass(t), expected)

    def test_renormalized_nonabstain_probs(self):
        t = tensor([[-1., -2., 0., 1.],
                    [-2., 0., 1., -1.]])
        expected = tensor([[0.2447, 0.0900, 0.6652],
                           [0.0351, 0.2595, 0.7054]])
        close_enough(renormalized_nonabstain_probs(t), expected)

    def test_gold_values(self):
        t = tensor([[0.2447, 0.0900, 0.6652],
                    [0.0351, 0.2595, 0.7054]])
        gold = tensor([1, 2])
        expected = tensor([0.0900, 0.7054])
        close_enough(gold_values(t, gold), expected)


if __name__ == "__main__":
    unittest.main()
