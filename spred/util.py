"""
Utility functions

"""
import torch
from torch.nn import functional

def close_enough(t1, t2):
    """whether two ```Tensors``` are close enough"""
    error_msg = "Not close enough:\n{}\n{}".format(t1, t2)
    assert torch.allclose(t1, t2, atol=0.001), error_msg


def softmax(t):
    """clamping then softmaxing, for numeric stability"""
    return functional.softmax(t.clamp(min=-25, max=25), dim=1)


def abstain_probs(output):
    """returns the abstention probability, corresponding to the abstention logit"""
    return softmax(output)[:, -1]


def nonabstain_probs(output):
    """returns the non abstention probabilities in an output"""
    return softmax(output)[:, 0:-1]


def nonabstain_prob_mass(output):
    """the sum of non-abstention probabilities"""
    return 1. - abstain_probs(output)


def renormalized_nonabstain_probs(output):
    """only normalize the non-abstention part of the probability distribution"""
    return softmax(output[:, 0:-1])


def gold_values(output, gold):
    """returns gold probabilities"""
    return output[list(range(output.shape[0])), gold]


def approx(x, y, num_sig=3):
    """whether two ```Tensors``` ```x``` and ```y``` are approximately equal,
    more lenient than ```closs_enough```"""
    return abs(x-y) < 1.0 * (10 ** (-num_sig))


class Logger:
    """For logging during experiments"""
    def __init__(self, verbose):
        self.verbose = verbose
    
    def __call__(self, msg):
        if self.verbose:
            print(msg)
