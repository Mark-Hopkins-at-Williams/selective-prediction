from abc import ABC, abstractmethod
import torch
from torch.nn import functional
from torch.autograd import Variable
from spred.util import cudaify


def softmax(t):
    return functional.softmax(t.clamp(min=-25, max=25), dim=1)


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


class DACLoss(SingleConfidenceLoss):

    def __init__(self, target_alpha, warmup_epochs, total_epochs, alpha_init_factor=64.):
        super(DACLoss, self).__init__()
        self.epsilon = 1e-7
        self.learn_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.alpha_final = target_alpha
        self.alpha_init_factor = alpha_init_factor
        self.alpha_var = None
        self.alpha_thresh_ewma = None  # exponentially weighted moving average for alpha_thresh
        self.alpha_thresh = None  # instantaneous alpha_thresh
        self.ewma_mu = 0.05  # mu parameter for EWMA;
        self.curr_alpha_factor = None  # for alpha initiliazation
        self.alpha_inc = None  # linear increase factor of alpha during abstention phase
        self.alpha_set_epoch = None
        self.vars = None
        self.epoch = 0

    def notify(self, e):
        self.epoch = e

    def _nll(self, output, gold):
        label_ps = output[list(range(len(output))), gold]
        label_ps = torch.clamp(label_ps, min=0.000000001)
        losses = -torch.log(label_ps)
        return losses

    def _h_c(self, output, gold):
        abs_prob = torch.clamp(output[:, -1], max=1. - self.epsilon)
        label_ps = output[list(range(len(output))), gold]
        true_class_prob = 1 - abs_prob
        true_class_prob = torch.clamp(true_class_prob, min=0.01)

        nan_tensor = torch.isnan(true_class_prob)
        assert (not (True in nan_tensor))

        normalized_true_prob = label_ps / true_class_prob
        normalized_true_prob = torch.clamp(normalized_true_prob, min=self.epsilon)
        thresholds = (- torch.log(normalized_true_prob))
        return thresholds

    def clip(self):
        if self.vars is not None:
            torch.nn.utils.clip_grad_norm(self.vars, max_norm=1)

    def __call__(self, input_batch, confidence, target_batch):
        if self.epoch <= self.learn_epochs:
            loss = functional.cross_entropy(input_batch, target_batch, reduction='none')
            h_c = functional.cross_entropy(input_batch[:, :-1], target_batch).detach()
            p_out = torch.exp(functional.log_softmax(input_batch, dim=1)).detach()
            p_out_abstain = p_out[:, -1].detach()
            # update instantaneous alpha_thresh
            self.alpha_thresh = Variable(((1. - p_out_abstain) * h_c).mean().data)
            # update alpha_thresh_ewma
            if self.alpha_thresh_ewma is None:
                self.alpha_thresh_ewma = self.alpha_thresh
            else:
                self.alpha_thresh_ewma = Variable(self.ewma_mu * self.alpha_thresh.data + \
                                                  (1. - self.ewma_mu) * self.alpha_thresh_ewma.data)
            return loss.mean()

        else:
            # calculate cross entropy only over true classes
            h_c = functional.cross_entropy(input_batch[:, 0:-1], target_batch, reduce='none')
            # probabilities of abstention  class
            p_out = torch.exp(functional.log_softmax(input_batch, dim=1))
            p_out_abstain = torch.min(p_out[:, -1],
                                      cudaify(Variable(torch.tensor([1. - self.epsilon]))))
            # update instantaneous alpha_thresh
            self.alpha_thresh = Variable(((1. - p_out_abstain) * h_c).mean().data)
            try:
                # update alpha_thresh_ewma
                if self.alpha_thresh_ewma is None:
                    self.alpha_thresh_ewma = self.alpha_thresh
                else:
                    self.alpha_thresh_ewma = Variable(self.ewma_mu * self.alpha_thresh.data + \
                                                      (1. - self.ewma_mu) * self.alpha_thresh_ewma.data)
                if self.alpha_var is None:  # hasn't been initialized. do it now
                    # we create a freshVariable here so that the history of alpha_var
                    # computation (which depends on alpha_thresh_ewma) is forgotten. This
                    # makes self.alpha_var a leaf variable, which will not be differentiated.
                    # aggressive initialization of alpha to jump start abstention
                    self.alpha_var = Variable(self.alpha_thresh_ewma.data / self.alpha_init_factor)
                    self.alpha_inc = (self.alpha_final - self.alpha_var.data) / (self.total_epochs - self.epoch)
                    self.alpha_set_epoch = self.epoch
                else:
                    # we only update alpha every epoch
                    if self.epoch > self.alpha_set_epoch:
                        self.alpha_var = Variable(self.alpha_var.data + self.alpha_inc)
                        self.alpha_set_epoch = self.epoch
                loss = (1. - p_out_abstain) * h_c - self.alpha_var * torch.log(1. - p_out_abstain)
                self.vars = [h_c, p_out_abstain]
                return loss.mean()
            except RuntimeError as e:
                print(e)

