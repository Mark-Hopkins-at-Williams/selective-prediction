from abc import ABC, abstractmethod
import torch
from torch.nn import functional
from torch.autograd import Variable
from spred.util import cudaify


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


class ConfidenceLoss(torch.nn.Module, ABC):
    def __init__(self):
        super(ConfidenceLoss, self).__init__()

    def notify(self, epoch):
        pass


class SingleConfidenceLoss(ConfidenceLoss):
    def __init__(self):
        super(SingleConfidenceLoss, self).__init__()

    @abstractmethod
    def __call__(self, output, confidence, gold):
        ...


class PairwiseConfidenceLoss(ConfidenceLoss):

    def __call__(self, output_x, output_y, gold_x, gold_y, conf_x, conf_y):
        def weighted_loss(weight_x, weight_y, loss_x, loss_y):
            weight_pair = torch.stack([weight_x, weight_y], dim=-1)
            softmaxed_weights = functional.softmax(weight_pair, dim=-1)
            loss_pair = torch.stack([loss_x, loss_y], dim=-1)
            return torch.sum(loss_pair * softmaxed_weights, dim=-1)

        loss = torch.nn.NLLLoss()
        nll_x = -loss(softmax(output_x), gold_x)
        nll_y = -loss(softmax(output_y), gold_y)
        losses = weighted_loss(conf_x, conf_y, nll_x, nll_y)
        return -torch.log(losses.mean())


class CrossEntropyLoss(SingleConfidenceLoss):
    """
    Does not assume that the output values are normalized.

    The CrossEntropyLoss performs its own softmax.

    """
    def __init__(self):
        super().__init__()
        self.loss = torch.nn.CrossEntropyLoss()

    def __call__(self, output, confidence, gold):
        return self.loss(output, gold)

    def __str__(self):
        return "CrossEntropyLoss"


class NLLLoss(SingleConfidenceLoss):
    def __init__(self):
        super().__init__()
        self.loss = torch.nn.NLLLoss()

    def __call__(self, output, confidence, gold):
        return self.loss(output, gold)

    def __str__(self):
        return "NLLLoss"


class DACLoss(SingleConfidenceLoss):

    def __init__(self, warmup_epochs, total_epochs):
        super(DACLoss, self).__init__()
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.epoch = 0
        self.loss_moving_average = None
        self.mu = 0.05
        self.rho = 64.0
        self.ce_loss = torch.nn.CrossEntropyLoss()

    def notify(self, e):
        self.epoch = e

    def __call__(self, output, confidence, gold):
        renormalized_gold_probs = gold_values(renormalized_nonabstain_probs(output),
                                              gold)
        term1 = nonabstain_prob_mass(output) * (-torch.log(renormalized_gold_probs))
        if self.epoch < self.warmup_epochs:
            t1 = term1.clone().detach()
            if self.loss_moving_average is None:
                self.loss_moving_average = torch.sum(t1)
            else:
                self.loss_moving_average = ((1 - self.mu) * self.loss_moving_average
                                            + self.mu * torch.sum(t1))
            multiplier = 0.0
            return torch.sum(term1)
        else:
            term2 = 1.0 / (1.0 - abstain_probs(output).clamp(max=0.9999))
            alpha = self.loss_moving_average / self.rho
            alpha_max = self.loss_moving_average
            discount = ((self.epoch - self.warmup_epochs) /
                        (self.total_epochs-self.warmup_epochs))
            multiplier = alpha + discount * (alpha_max - alpha)
            return torch.sum(term1 + multiplier * term2)


class CrossEntropyLossWithErrorRegularization(SingleConfidenceLoss):
    """
    TODO: test the gradients and make sure they work properly.

    """
    def __init__(self, lambda_param):
        super().__init__()
        self.lambda_param = lambda_param
        self.loss = torch.nn.CrossEntropyLoss()

    def __call__(self, output, confidence, gold):
        ce_loss = self.loss(output, gold)
        probs = softmax(output)
        truthvals = (probs.max(dim=1).indices == gold)
        correct_confs = confidence[truthvals]
        incorrect_confs = confidence[~truthvals]
        diffs = (incorrect_confs.unsqueeze(1) - correct_confs.unsqueeze(0)).view(-1)
        penalty = torch.sum(torch.clamp(diffs, min=0)**2)
        # ALTERNATIVE USING LOGS
        # correct_confs = -torch.log(confidence[truthvals])
        # incorrect_confs = -torch.log(confidence[~truthvals])
        # diffs = (correct_confs.unsqueeze(1) - incorrect_confs.unsqueeze(0)).view(-1)
        # penalty = torch.mean(torch.clamp(diffs, min=0))
        return ce_loss + self.lambda_param * penalty

    def __str__(self):
        return "CrossEntropyLoss"


class AbstainingLoss(SingleConfidenceLoss):
    def __init__(self, alpha=0.5, warmup_epochs=3):
        super().__init__()
        self.alpha = 0.0
        self.warmup_epochs = warmup_epochs
        self.target_alpha = alpha
        self.notify(0)

    def notify(self, epoch):
        if epoch >= self.warmup_epochs:
            self.alpha = self.target_alpha

    def __call__(self, output, confidence, gold):
        dists = softmax(output)
        label_ps = dists[list(range(output.shape[0])), gold]
        abstains = dists[:, -1]
        losses = label_ps + (self.alpha * abstains)
        losses = torch.clamp(losses, min=0.000000001)
        return -torch.mean(torch.log(losses))

    def __str__(self):
        return "AbstainingLoss_target_alpha_" + str(self.target_alpha)

