import torch
from torch.nn import functional

def close_enough(t1, t2):
    error_msg = "Not close enough:\n{}\n{}".format(t1, t2)
    assert torch.allclose(t1, t2, atol=0.001), error_msg


def softmax(t):
    return functional.softmax(t.clamp(min=-25, max=25), dim=1)


def abstain_probs(output):
    return softmax(output)[:, -1]


def nonabstain_probs(output):
    return softmax(output)[:, 0:-1]


def nonabstain_prob_mass(output):
    return 1. - abstain_probs(output)


def renormalized_nonabstain_probs(output):
    return softmax(output[:, 0:-1])


def gold_values(output, gold):
    return output[list(range(output.shape[0])), gold]


def approx(x, y, num_sig=3):
    return abs(x-y) < 1.0 * (10 ** (-num_sig))


class Logger:
    def __init__(self, verbose):
        self.verbose = verbose
    
    def __call__(self, msg):
        if self.verbose:
            print(msg)
