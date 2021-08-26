import torch
import torch.nn.functional as F
from spred.util import cudaify
from torch.autograd import Variable
from spred.loss import SingleConfidenceLoss

epsilon = 1e-7


class DACLoss(SingleConfidenceLoss):

    def __init__(self, target_alpha, warmup_epochs, total_epochs, alpha_init_factor=64.):
        super(DACLoss, self).__init__()
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
        abs_prob = torch.clamp(output[:, -1], max=1. - epsilon)
        label_ps = output[list(range(len(output))), gold]
        true_class_prob = 1 - abs_prob
        true_class_prob = torch.clamp(true_class_prob, min=0.01)

        nan_tensor = torch.isnan(true_class_prob)
        assert (not (True in nan_tensor))

        normalized_true_prob = label_ps / true_class_prob
        normalized_true_prob = torch.clamp(normalized_true_prob, min=epsilon)
        thresholds = (- torch.log(normalized_true_prob))
        return thresholds

    def clip(self):
        if self.vars is not None:
            torch.nn.utils.clip_grad_norm(self.vars, max_norm=1)

    def __call__(self, input_batch, confidence, target_batch):
        if self.epoch <= self.learn_epochs:
            loss = F.cross_entropy(input_batch, target_batch, reduction='none')
            h_c = F.cross_entropy(input_batch[:, :-1], target_batch).detach()
            p_out = torch.exp(F.log_softmax(input_batch, dim=1)).detach()
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
            h_c = F.cross_entropy(input_batch[:, 0:-1], target_batch, reduce='none')
            # probabilities of abstention  class
            p_out = torch.exp(F.log_softmax(input_batch, dim=1))
            p_out_abstain = torch.min(p_out[:, -1],
                                      cudaify(Variable(torch.tensor([1. - epsilon]))))
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

