import unittest
from math import log
import torch
from torch import tensor
from spred.loss import NLLLoss, PairwiseConfidenceLoss, CrossEntropyLoss
from spred.loss import AbstainingLoss, ConfidenceLoss4
from torch.nn import functional


def approx(x, y, num_digits=4):
    return abs(x-y) < 1.0 * (10 ** -num_digits)


def softmax(t):
    return functional.softmax(t.clamp(min=-25, max=25), dim=1)


def close_enough(t1, t2):
    error_msg = "Not close enough:\n{}\n{}".format(t1, t2)
    assert torch.allclose(t1, t2, atol=0.001), error_msg


class TestLoss(unittest.TestCase):
    def test_nll_loss1(self):
        predictions = tensor([[-1., -2., -3.]])
        loss_function = NLLLoss()
        gold = torch.tensor([2])
        loss = loss_function(predictions, None, gold)
        close_enough(loss, tensor(3.))

    def test_nll_loss2(self):
        predictions = tensor([[-1., -2., -3.],
                              [4., 5., 6.]])
        loss_function = NLLLoss()
        gold = torch.tensor([2, 1])
        loss = loss_function(predictions, None, gold)
        close_enough(loss, tensor((3-5)/2.))

    def test_cross_entropy_loss1(self):
        predictions = tensor([[-1., -2., -3.]])
        predicted_probs = softmax(predictions)
        close_enough(predicted_probs, tensor([[0.6652, 0.2447, 0.0900]]))
        loss_function = CrossEntropyLoss()
        gold = torch.tensor([2])
        loss = loss_function(predictions, None, gold)
        close_enough(loss, tensor(-log(.0900)))

    def test_cross_entropy_loss2(self):
        predictions = tensor([[-1., -2., -3.],
                              [-1., -2., -3.]])
        predicted_probs = softmax(predictions)
        close_enough(predicted_probs, tensor([[0.6652, 0.2447, 0.0900],
                                              [0.6652, 0.2447, 0.0900]]))
        loss_function = CrossEntropyLoss()
        gold = torch.tensor([2, 1])
        loss = loss_function(predictions, None, gold)
        close_enough(loss, tensor((-log(.0900) - log(.2447))/2))

    def test_abstaining_loss1(self):
        predictions = tensor([[-1., -2., 0., 1.]])
        predicted_probs = softmax(predictions)
        close_enough(predicted_probs, tensor([[0.0871, 0.0321, 0.2369, 0.6439]]))
        gold = torch.tensor([2])
        alpha = 0.5
        loss_function = AbstainingLoss(alpha=alpha, warmup_epochs=3)
        loss_function.notify(0)
        loss = loss_function(predictions, None, gold)
        close_enough(loss, tensor(-log(0.2369)))
        loss_function.notify(4)
        loss = loss_function(predictions, None, gold)
        expected = -log(0.2369 + alpha * 0.6439)
        close_enough(loss, tensor(expected))

    def test_abstaining_loss2(self):
        predictions = tensor([[-1., -2., 0., 1.], [-2., 0., 1., -1.]])
        predicted_probs = softmax(predictions)
        close_enough(predicted_probs, tensor([[0.0871, 0.0321, 0.2369, 0.6439],
                                              [0.0321, 0.2369, 0.6439, 0.0871]]))
        gold = torch.tensor([2, 0])
        alpha = 0.5
        loss_function = AbstainingLoss(alpha=alpha, warmup_epochs=3)
        loss_function.notify(4)
        loss = loss_function(predictions, None, gold)
        expected = (-log(0.2369 + alpha * 0.6439) +
                    -log(0.0321 + alpha * 0.0871)) / 2
        close_enough(loss, tensor(expected))

    def test_conf4_1(self):
        predictions = tensor([[-1., -2., 0., 1.]])
        predicted_probs = softmax(predictions)
        predicted_probs_no_abstain = softmax(predictions[:, :-1])
        close_enough(predicted_probs, tensor([[0.0871, 0.0321, 0.2369, 0.6439]]))
        close_enough(predicted_probs_no_abstain, tensor([[0.2447, 0.0900, 0.6652]]))
        gold = torch.tensor([2])
        alpha = 0.5
        loss_function = ConfidenceLoss4(alpha=alpha, warmup_epochs=3)
        loss_function.notify(4)
        loss = loss_function(predictions, None, gold)
        expected = -log(0.6652 * (0.2369 + alpha * 0.6439))
        close_enough(loss, tensor(expected))

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


if __name__ == "__main__":
    unittest.main()
