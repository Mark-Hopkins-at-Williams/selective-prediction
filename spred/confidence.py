import torch
from torch.nn import functional
from spred.loss import softmax, gold_values


def inv_abstain_prob(_, output):
    probs = functional.softmax(output.clamp(min=-25, max=25), dim=-1)
    return 1.0 - probs[:, -1]


def max_nonabstain_prob(_, output):
    probs = functional.softmax(output.clamp(min=-25, max=25), dim=-1)
    return probs[:, :-1].max(dim=1).values


def max_prob(_, output):
    probs = functional.softmax(output.clamp(min=-25, max=25), dim=-1)
    return probs.max(dim=1).values


def abstention(_, output):
    return output[:, -1]


def random_confidence(_, output):
    return torch.randn(output.shape[0])


def lookup_confidence_extractor(name, model):
    confidence_extractor_lookup = {'inv_abs': inv_abstain_prob,
                                   'max_non_abs': max_nonabstain_prob,
                                   'abs': abstention,
                                   'max_prob': max_prob,
                                   'random': random_confidence}
    if name in confidence_extractor_lookup:
        return confidence_extractor_lookup[name]
    elif name == 'mc_dropout':
        return MCDropoutConfidence(model)
    else:
        raise Exception('Confidence extractor not recognized: {}'.format(name))


class MCDropoutConfidence:
    def __init__(self, model, n_forward_passes=30):
        self.model = model
        self.n_forward_passes = n_forward_passes

    def __call__(self, input, output):
        preds = torch.max(output, dim=1).indices
        pred_probs = []
        for _ in range(self.n_forward_passes):
            self.model.train()
            dropout_output, _ = self.model(input, compute_conf=False)
            dropout_output = softmax(dropout_output)
            pred_probs.append(gold_values(dropout_output, preds))
        pred_probs = torch.stack(pred_probs)
        confs = torch.mean(pred_probs, dim=0)
        return confs
