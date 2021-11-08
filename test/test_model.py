import unittest
from math import log
import torch
from torch import tensor
from spred.loss import CrossEntropyLoss
from spred.model import Feedforward
from spred.util import softmax, close_enough
from torch.nn import functional
from spred.confidence import MaxProb
from test.examples import build_interface_a_net, build_interface_b_net
from test.examples import ExampleFeedforwardLoader

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
                            incl_abstain=False)
        model.set_confidence_extractor(MaxProb())
        batch = example_batch1()
        model_out = model(batch, compute_conf=True)
        assert model_out['outputs'].shape == (2, 3)
        assert model_out['confidences'].shape == (2, )

    def test_interface_a(self):
        model = build_interface_a_net()
        loader = ExampleFeedforwardLoader()
        model_out = model(next(iter(loader)))
        expected = {'outputs': tensor([[-0.0820,  0.1540],
                                       [-0.0530,  0.2970]]),
                    'loss': tensor(0.73273,),
                    'confidences': tensor([0.5587, 0.5866])}
        close_enough(model_out['outputs'], expected['outputs'])
        close_enough(model_out['loss'], expected['loss'])
        close_enough(model_out['confidences'], expected['confidences'])

    def test_interface_b(self):
        model = build_interface_b_net()
        loader = ExampleFeedforwardLoader()
        model_out = model(next(iter(loader)))
        expected = {'outputs': tensor([[0.1180, 0.2540, 0.3900],
                                       [0.1540, 0.3970, 0.6330]]),
                    'loss': tensor(1.2315),
                    'confidences': tensor([0.3313, 0.3278])}
        close_enough(model_out['outputs'], expected['outputs'])
        close_enough(model_out['loss'], expected['loss'])
        close_enough(model_out['confidences'], expected['confidences'])


if __name__ == "__main__":
    unittest.main()
