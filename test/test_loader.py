import unittest
import torch
from torch import tensor
from spred.tasks.normals import NormalsLoader

def approx(x, y, num_digits=4):
    return abs(x-y) < 1.0 * (10 ** -num_digits)


def softmax(t):
    return functional.softmax(t.clamp(min=-25, max=25), dim=1)


def close_enough(t1, t2):
    error_msg = "Not close enough:\n{}\n{}".format(t1, t2)
    assert torch.allclose(t1, t2, atol=0.001), error_msg


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
