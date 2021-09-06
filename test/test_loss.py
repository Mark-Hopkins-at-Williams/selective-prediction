import unittest
from math import log
import torch
from torch import tensor
from spred.loss import NLLLoss, PairwiseConfidenceLoss, CrossEntropyLoss
from spred.loss import LossWithErrorRegularization
from spred.loss import AbstainingLoss, DACLoss
from spred.util import softmax, close_enough
from torch.nn import functional


def example_batch1():
    outputs = tensor([[-1., -2., -3.],
                      [-2., -1., -2.5],
                      [-1., -3.5, -2.],
                      [-2., -3., -1.5],
                      [-3., -0.5, -2.],
                      [-3.5, -3, -1.]])
    labels = torch.tensor([0, 0, 1, 2, 1, 1])
    confs = softmax(outputs).max(dim=1).values
    close_enough(softmax(outputs), tensor([[0.6652, 0.2447, 0.0900],
                                           [0.2312, 0.6285, 0.1402],
                                           [0.6897, 0.0566, 0.2537],
                                           [0.3315, 0.1220, 0.5465],
                                           [0.0629, 0.7662, 0.1710],
                                           [0.0674, 0.1112, 0.8214]]))
    close_enough(confs, tensor([0.6652, 0.6285, 0.6897, 0.5465, 0.7662, 0.8214]))
    batch = {'outputs': outputs, 'labels': labels, 'confidences': confs}
    return batch

def example_batch2():
    outputs = tensor([[-1., -2., 0., 1.],
                      [-2., 0., 1., -1.]])
    labels = torch.tensor([2, 0])
    confs = softmax(outputs).max(dim=1).values
    close_enough(softmax(outputs), tensor([[0.0871, 0.0321, 0.2369, 0.6439],
                                           [0.0321, 0.2369, 0.6439, 0.0871]]))
    close_enough(confs, tensor([0.6439, 0.6439]))
    batch = {'outputs': outputs, 'labels': labels, 'confidences': confs}
    return batch


class TestLoss(unittest.TestCase):

    def test_cross_entropy_loss(self):
        loss_fn = CrossEntropyLoss()
        batch = example_batch2()
        loss = loss_fn(batch)
        close_enough(loss, tensor((-log(0.2369) - log(0.0321))/2))

    def test_error_reg_loss(self):
        base_loss_fn = CrossEntropyLoss()
        loss_fn = LossWithErrorRegularization(base_loss_fn, lambda_param=0.5)
        batch = example_batch1()
        loss = loss_fn(batch)
        base_loss = base_loss_fn(batch)
        expected_penalty = ((.6652-.6897)**2 + (.6652-.8214)**2 + (.5465-.6285)**2 +
                            (.5465-.6897)**2 + (.5465-.8214)**2 + (.7662-.8214)**2)
        close_enough(loss, base_loss + 0.5 * expected_penalty)

    def test_abstaining_loss(self):
        batch = example_batch2()
        alpha = 0.5
        loss_function = AbstainingLoss(alpha=alpha, warmup_epochs=3)
        loss_function.notify(4)
        loss = loss_function(batch)
        expected = (-log(0.2369 + alpha * 0.6439) +
                    -log(0.0321 + alpha * 0.0871)) / 2
        close_enough(loss, tensor(expected))

    """
    def test_dac_loss(self):
        loss_function = DACLoss(warmup_epochs=5,
                                total_epochs=15, alpha_init_factor=64.0)
        predictions = tensor([[-1., -2., 0., 1.],
                              [-2., 0., 1., -1.]])
        close_enough(softmax(predictions), tensor([[0.0871, 0.0321, 0.2369, 0.6439],
                                                   [0.0321, 0.2369, 0.6439, 0.0871]]))
        gold = torch.tensor([2, 0])
        loss_function.notify(8)
        loss = loss_function(predictions, None, gold)
        # expected = (-log(0.2369 + alpha * 0.6439) +
        #             -log(0.0321 + alpha * 0.0871)) / 2
        ce = CrossEntropyLoss()
        ce_loss = ce(predictions, None, gold)
        # print("DAC Loss: {}".format(loss))
        # print("CE Loss:  {}".format(ce_loss))
        # close_enough(loss, tensor(expected))

    def test_pairwise1(self):
        predictions_a = tensor([[-1., -2., -3.]])
        gold_a = torch.tensor([1])
        close_enough(softmax(predictions_a), tensor([[0.6652, 0.2447, 0.0900]]))
        predictions_b = tensor([[0., -1., 3.]])
        gold_b = torch.tensor([2])
        close_enough(softmax(predictions_b), tensor([[0.0466, 0.0171, 0.9362]]))
        conf_a = torch.tensor([-0.3])
        conf_b = torch.tensor([1.8])
        close_enough(softmax(tensor([[-0.3, 1.8]])), tensor([[0.1091, 0.8909]]))
        loss_function = PairwiseConfidenceLoss()
        loss = loss_function(predictions_a, predictions_b,
                             gold_a, gold_b, conf_a, conf_b)
        expected_loss = tensor(-log(0.1091 * 0.2447 + 0.8909 * 0.9362))
        assert (torch.allclose(expected_loss, loss, atol=0.0001))

    def test_pairwise2(self):
        predictions_a = tensor([[-1., -2., -3.], [-0.4, 1.1, 0.3]])
        gold_a = torch.tensor([1, 2])
        close_enough(softmax(predictions_a), tensor([[0.6652, 0.2447, 0.0900],
                                                     [0.1334, 0.5979, 0.2687]]))
        predictions_b = tensor([[0., -1., 3.], [0.3, -0.7, -1.2]])
        gold_b = torch.tensor([2, 0])
        close_enough(softmax(predictions_b), tensor([[0.0466, 0.0171, 0.9362],
                                                     [0.6285, 0.2312, 0.1402]]))
        conf_a = torch.tensor([-0.3, 0.5])
        conf_b = torch.tensor([1.8, 0.1])
        close_enough(softmax(tensor([[-0.3, 1.8]])), tensor([[0.1091, 0.8909]]))
        close_enough(softmax(tensor([[0.5, 0.1]])), tensor([[0.5987, 0.4013]]))
        loss_function = PairwiseConfidenceLoss()
        loss = loss_function(predictions_a, predictions_b,
                             gold_a, gold_b, conf_a, conf_b)
        expected_loss = tensor(0.5 * -(log(0.1091 * 0.2447 + 0.8909 * 0.9362)
                                       + log(0.5987 * 0.2687 + 0.4013 * 0.6285)))
        assert (torch.allclose(expected_loss, loss, atol=0.0001))
    """

if __name__ == "__main__":
    unittest.main()
