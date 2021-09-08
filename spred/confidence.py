import torch
from torch.nn import functional
from spred.loss import softmax, gold_values


def lookup_confidence_extractor(name, model):
    confidence_extractor_lookup = {'inv_abstain': inv_abstain_prob,
                                   'max_non_abstain': max_nonabstain_prob,
                                   'max_prob': max_prob,
                                   'random': random_confidence,
                                   'normals_gold': normals_gold_conf}
    if name in confidence_extractor_lookup:
        return confidence_extractor_lookup[name]
    elif name == 'mc_dropout':
        return MCDropoutConfidence(model)
    else:
        raise Exception('Confidence extractor not recognized: {}'.format(name))


def inv_abstain_prob(batch):
    output = batch['outputs']
    probs = functional.softmax(output.clamp(min=-25, max=25), dim=-1)
    return 1.0 - probs[:, -1]


def max_nonabstain_prob(batch):
    output = batch['outputs']
    probs = functional.softmax(output.clamp(min=-25, max=25), dim=-1)
    return probs[:, :-1].max(dim=1).values


def max_prob(batch):
    output = batch['outputs']
    probs = functional.softmax(output.clamp(min=-25, max=25), dim=-1)
    return probs.max(dim=1).values


def random_confidence(batch):
    output = batch['outputs']
    return torch.rand(output.shape[0])


def normals_gold_conf(batch):
    # TRY AGAIN: this isn't the gold confidence for normals
    inputs = batch['inputs']['inputs']
    inputs = inputs[:,:2]
    adjusted = inputs**2
    adjusted = torch.stack([adjusted[:,0] / 0.5, adjusted[:,1] / 0.1])
    return adjusted.sum(dim=0)


class MCDropoutConfidence:
    def __init__(self, model, n_forward_passes=30, combo_fn=torch.mean):
        self.model = model
        self.n_forward_passes = n_forward_passes
        self.combo_fn = combo_fn

    def __call__(self, batch):
        input = batch['inputs']
        output = batch['outputs']
        preds = torch.max(output, dim=1).indices
        pred_probs = []
        for _ in range(self.n_forward_passes):
            self.model.train()
            model_out = self.model(input, compute_conf=False)
            dropout_output = softmax(model_out['outputs'])
            pred_probs.append(gold_values(dropout_output, preds))
        pred_probs = torch.stack(pred_probs)
        confs = self.combo_fn(pred_probs, dim=0)
        return confs
