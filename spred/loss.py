import torch
from torch.nn import functional
from spred.util import abstract_method


def softmax(t):
    return functional.softmax(t.clamp(min=-25, max=25), dim=1)


class ConfidenceLoss(torch.nn.Module):
    def __init__(self):
        super(ConfidenceLoss, self).__init__()

    def notify(self, epoch):
        pass


class SingleConfidenceLoss(ConfidenceLoss):
    def __init__(self):
        super(SingleConfidenceLoss, self).__init__()

    def __call__(self, output, confidence, gold):
        abstract_method()


class CrossEntropyLoss(SingleConfidenceLoss):
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


class ConfidenceLoss4(SingleConfidenceLoss):
    def __init__(self, alpha=0.5, warmup_epochs=5):
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
        initial_losses = (label_ps + (self.alpha * abstains))
        label_ps_woa = softmax(output[:, :-1])
        label_ps_woa = label_ps_woa[list(range(label_ps_woa.shape[0])), gold]
        losses = label_ps_woa * initial_losses
        losses = torch.clamp(losses, min=0.000000001)
        return -torch.mean(torch.log(losses))

    def __str__(self):
        return "ConfidenceLoss4_p0_" + str(self.p0)


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
