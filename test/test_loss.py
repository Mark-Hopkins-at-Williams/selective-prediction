import unittest
from math import log
import torch
from torch import tensor
from spred.loss import CrossEntropyLoss
from spred.loss import ErrorRegularizer
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
    batch = {'outputs': outputs, 'labels': labels, 'confidences': confs, 'loss': 1.0}
    return batch

def example_batch2():
    outputs = tensor([[-1., -2., 0., 1.],
                      [-2., 0., 1., -1.]])
    labels = torch.tensor([2, 0])
    confs = softmax(outputs).max(dim=1).values
    close_enough(softmax(outputs), tensor([[0.0871, 0.0321, 0.2369, 0.6439],
                                           [0.0321, 0.2369, 0.6439, 0.0871]]))
    close_enough(confs, tensor([0.6439, 0.6439]))
    batch = {'outputs': outputs, 'labels': labels, 'confidences': confs, 'loss': 1.0}
    return batch

def example_batch3():
    outputs = tensor([[-1., -2.],
                      [3., -1.],
                      [-3., 3.],
                      [-1.5, 2.]])
    labels = torch.tensor([1, 0, 1, 0])  # F T T F
    confs = torch.tensor([0.2, 0.3, 0.4, 0.5], requires_grad=True)
    batch = {'outputs': outputs, 'labels': labels, 'confidences': confs, 'loss': 1.0}
    return confs, batch


class TestLoss(unittest.TestCase):

    def test_cross_entropy_loss(self):
        loss_fn = CrossEntropyLoss()
        batch = example_batch2()
        loss = loss_fn(batch)
        close_enough(loss, tensor((-log(0.2369) - log(0.0321))/2))

    def test_error_reg_loss(self):
        loss_fn = ErrorRegularizer(lambda_param=0.5)
        batch = example_batch1()
        loss = loss_fn(batch)
        base_loss = torch.tensor(1.0)
        expected_penalty = ((.6652-.6897)**2 + (.6652-.8214)**2 + (.5465-.6285)**2 +
                            (.5465-.6897)**2 + (.5465-.8214)**2 + (.7662-.8214)**2)
        close_enough(loss, base_loss + 0.5 * expected_penalty)

    def test_error_reg_loss_grad(self):
        loss_fn = ErrorRegularizer(lambda_param=0.5)
        confs, batch = example_batch3()
        loss = loss_fn(batch)
        expected_penalty = ((.3-.5)**2 + (.4-.5)**2)
        close_enough(loss, tensor(1.0 + 0.5 * expected_penalty))
        loss.backward()
        close_enough(confs.grad, tensor([ 0.0, -0.2, -0.1,  0.3]))

    def test_abstaining_loss(self):
        batch = example_batch2()
        alpha = 0.5
        loss_function = AbstainingLoss(alpha=alpha, warmup_epochs=3)
        loss_function.notify(4)
        loss = loss_function(batch)
        expected = (-log(alpha * 0.6439) +
                    -log(alpha * 0.0871)) / 2
        close_enough(loss, tensor(1.0 + expected))

    def test_dac_loss(self):
        batch = example_batch2()
        loss_function = DACLoss(warmup_epochs=5)
        ce = CrossEntropyLoss()
        ce_loss = ce(batch)
        loss_function.notify(3)
        loss = loss_function(batch)
        close_enough(loss, ce_loss)
        loss_function.notify(8)
        loss = loss_function(batch)
        close_enough(loss, tensor(1.615211))


if __name__ == "__main__":
    unittest.main()
