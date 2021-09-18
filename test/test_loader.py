import unittest
import torch
from torch import tensor
from spred.tasks.normals import NormalsLoader
from spred.loader import Loader, BalancedLoader

def approx(x, y, num_digits=4):
    return abs(x-y) < 1.0 * (10 ** -num_digits)


def softmax(t):
    return functional.softmax(t.clamp(min=-25, max=25), dim=1)


def close_enough(t1, t2):
    error_msg = "Not close enough:\n{}\n{}".format(t1, t2)
    assert torch.allclose(t1, t2, atol=0.001), error_msg


class ExampleLoader(Loader):

    def __init__(self):
        batch1 = {'inputs': tensor([[0, 0.5], [1, 1.5], [2, 2.5],
                                    [3, 3.5], [4, 4.5], [5, 5.5]]),
                  'labels': tensor([0,1,0,1,1,1])}
        batch2 = {'inputs': tensor([[6, 6.5], [7, 7.5], [8, 8.5],
                                    [9, 9.5], [10, 10.5], [11, 11.5]]),
                  'labels': tensor([0, 1, 0, 0, 0, 0])}
        self.batches = [batch1, batch2]

    def __iter__(self):
        for batch in self.batches:
            yield batch

    def __len__(self):
        return 3

    def input_size(self):
        return 2

    def output_size(self):
        return 2


class TestLoader(unittest.TestCase):
    def test_balanced_loader(self):
        base_loader = ExampleLoader()
        loader = BalancedLoader(base_loader)
        ... # TODO: complete!



class TestNormalsLoader(unittest.TestCase):

    def test_abstain_probs(self):
        loader = NormalsLoader(2, 4, 3)
        loader_iter = iter(loader)
        batch1 = next(loader_iter)
        batch2 = next(loader_iter)
        assert batch1['inputs'].shape == (4, 5)
        assert batch1['labels'].shape == (4, )
        assert batch2['inputs'].shape == (4, 5)
        assert batch2['labels'].shape == (4,)
        try:
            next(loader_iter)
            assert False, "shouldn't have 3 batches"
        except StopIteration:
            pass



if __name__ == "__main__":
    unittest.main()
