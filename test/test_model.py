import unittest
from math import log
import torch
from torch import tensor
from spred.loss import CrossEntropyLoss
from spred.model import Feedforward
from spred.util import softmax, close_enough
from torch.nn import functional


def example_batch1():
    inputs = tensor([[-1., -2., 0, 3, -1.5],
                     [-2., 0., 1., 2, -1]])
    labels = torch.tensor([2, 0])
    batch = {'inputs': inputs, 'labels': labels}
    return batch


class TestModel(unittest.TestCase):

    def test_feedforward(self):
        model = Feedforward(input_size=5, hidden_sizes=(7, 7), output_size=3,
                            loss_f=CrossEntropyLoss(),
                            confidence_extractor='max_prob')
        batch = example_batch1()
        model_out = model(batch, compute_conf=True)
        assert model_out['outputs'].shape == (2, 3)
        assert model_out['confidences'].shape == (2, )


if __name__ == "__main__":
    unittest.main()
