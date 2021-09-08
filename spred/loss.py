from abc import ABC, abstractmethod
import torch
from torch.nn import functional
from torch.autograd import Variable
import pdb
import math
from spred.util import abstain_probs, nonabstain_probs
from spred.util import renormalized_nonabstain_probs
from spred.util import nonabstain_prob_mass, gold_values, softmax

def init_loss_fn(config):
    loss_config = config['trainer']['loss']
    loss_lookup = {'crossentropy': CrossEntropyLoss,
                   'ereg': LossWithErrorRegularization,
                   'conf1': AbstainingLoss,
                   'pairwise': PairwiseConfidenceLoss,
                   'dac': DACLoss}
    params = {k: v for k, v in loss_config.items() if k != 'name'}
    if loss_config['name'] == 'dac':
        params['total_epochs'] = config['trainer']['n_epochs']
    return loss_lookup[loss_config['name']](**params)


class ConfidenceLoss(torch.nn.Module, ABC):
    def __init__(self):
        super(ConfidenceLoss, self).__init__()

    @abstractmethod
    def __call__(self, batch):
        ...

    def notify(self, epoch):
        pass


class CrossEntropyLoss(ConfidenceLoss):
    """
    Does not assume that the output values are normalized.

    The CrossEntropyLoss performs its own softmax.

    """
    def __init__(self):
        super().__init__()
        self.loss = torch.nn.CrossEntropyLoss()

    def __call__(self, batch):
        return self.loss(batch['outputs'], batch['labels'])

    def __str__(self):
        return "CrossEntropyLoss"


class LossWithErrorRegularization(ConfidenceLoss):
    """
    TODO: test the gradients and make sure they work properly.

    """
    def __init__(self, base_loss, lambda_param):
        super().__init__()
        self.lambda_param = lambda_param
        self.base_loss = init_loss_fn(base_loss)

    def __call__(self, batch):
        output, gold, confidence = batch['outputs'], batch['labels'], batch['confidences']
        ce_loss = self.base_loss(batch)
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
        return "LossWithErrorRegularization"


class AbstainingLoss(ConfidenceLoss):
    # TODO: rethink this loss function
    def __init__(self, alpha=0.5, warmup_epochs=3):
        super().__init__()
        self.alpha = 0.0
        self.warmup_epochs = warmup_epochs
        self.target_alpha = alpha
        self.notify(0)

    def notify(self, epoch):
        if epoch >= self.warmup_epochs:
            self.alpha = self.target_alpha

    def __call__(self, batch):
        output, gold, confidence = batch['outputs'], batch['labels'], batch['confidences']
        dists = softmax(output)
        label_ps = dists[list(range(output.shape[0])), gold]
        abstains = dists[:, -1]
        losses = label_ps + (self.alpha * abstains)
        losses = torch.clamp(losses, min=0.000000001)
        return -torch.mean(torch.log(losses))

    def __str__(self):
        return "AbstainingLoss_target_alpha_" + str(self.target_alpha)


class DACLoss(ConfidenceLoss):
    # for numerical stability
    epsilon = 1e-7

    def __init__(self, warmup_epochs, total_epochs,
                 use_cuda=False, cuda_device=None,
                 alpha_final=1.0, alpha_init_factor=64.):
        super(ConfidenceLoss, self).__init__()
        self.learn_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.alpha_final = alpha_final
        self.alpha_init_factor = alpha_init_factor
        self.use_cuda = use_cuda
        self.cuda_device = cuda_device
        self.alpha_var = None
        self.alpha_thresh_ewma = None   #exponentially weighted moving average for alpha_thresh
        self.alpha_thresh = None #instantaneous alpha_thresh
        self.ewma_mu = 0.05 #mu parameter for EWMA;
        self.curr_alpha_factor  = None #for alpha initiliazation
        self.alpha_inc = None #linear increase factor of alpha during abstention phase
        self.alpha_set_epoch = None
        self.epoch = None
        self.in_training = True
        self.notify(0)

    def notify(self, epoch):
        self.epoch = epoch

    def __call__(self, batch):
        input_batch, target_batch = batch['outputs'], batch['labels']
        if self.epoch <= self.learn_epochs or not self.in_training:
            loss =  functional.cross_entropy(input_batch, target_batch, reduction='none')
            if self.in_training:
                h_c = functional.cross_entropy(input_batch[:,0:-1],target_batch,reduction='none')
                p_out = torch.exp(functional.log_softmax(input_batch,dim=1))
                p_out_abstain = p_out[:,-1]

                #update instantaneous alpha_thresh
                self.alpha_thresh = Variable(((1. - p_out_abstain)*h_c).mean().data)
                #update alpha_thresh_ewma
                if self.alpha_thresh_ewma is None:
                    self.alpha_thresh_ewma = self.alpha_thresh
                else:
                    self.alpha_thresh_ewma = Variable(self.ewma_mu*self.alpha_thresh.data + \
                        (1. - self.ewma_mu)*self.alpha_thresh_ewma.data)
            return loss.mean()

        else:
            #calculate cross entropy only over true classes
            h_c = functional.cross_entropy(input_batch[:,0:-1],target_batch,reduce=False)
            p_out = torch.exp(functional.log_softmax(input_batch,dim=1))
            #probabilities of abstention  class
            p_out_abstain = p_out[:,-1]

            # avoid numerical instability by upper-bounding
            # p_out_abstain to never be more than  1 - eps since we have to
            # take log(1 - p_out_abstain) later.
            if self.use_cuda:
                p_out_abstain = torch.min(p_out_abstain,
                    Variable(torch.Tensor([1. - epsilon])).cuda(self.cuda_device))
            else:
                p_out_abstain = torch.min(p_out_abstain,
                    Variable(torch.Tensor([1. - epsilon])))

            #update instantaneous alpha_thresh
            self.alpha_thresh = Variable(((1. - p_out_abstain)*h_c).mean().data)
            try:
                #update alpha_thresh_ewma
                if self.alpha_thresh_ewma is None:
                    self.alpha_thresh_ewma = self.alpha_thresh #Variable(((1. - p_out_abstain)*h_c).mean().data)
                else:
                    self.alpha_thresh_ewma = Variable(self.ewma_mu*self.alpha_thresh.data + \
                        (1. - self.ewma_mu)*self.alpha_thresh_ewma.data)


                if self.alpha_var is None: #hasn't been initialized. do it now
                    #we create a freshVariable here so that the history of alpha_var
                    #computation (which depends on alpha_thresh_ewma) is forgotten. This
                    #makes self.alpha_var a leaf variable, which will not be differentiated.
                    #aggressive initialization of alpha to jump start abstention
                    self.alpha_var = Variable(self.alpha_thresh_ewma.data /self.alpha_init_factor)
                    self.alpha_inc = (self.alpha_final - self.alpha_var.data)/(self.total_epochs - epoch)
                    self.alpha_set_epoch = epoch

                else:
                    # we only update alpha every epoch
                    if self.epoch > self.alpha_set_epoch:
                        self.alpha_var = Variable(self.alpha_var.data + self.alpha_inc)
                        self.alpha_set_epoch = self.epoch

                loss = ((1. - p_out_abstain)*h_c -
                        self.alpha_var*torch.log(1. - p_out_abstain))
                return loss.mean()
            except RuntimeError as e:
                print(e)


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
