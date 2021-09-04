import torch
from torch import nn
import torch.nn.functional as F
from spred.util import cudaify
from spred.loss import softmax, gold_values
from transformers import AutoModelForSequenceClassification


def inv_abstain_prob(_, output):
    probs = F.softmax(output.clamp(min=-25, max=25), dim=-1)
    return 1.0 - probs[:, -1]


def max_nonabstain_prob(_, output):
    probs = F.softmax(output.clamp(min=-25, max=25), dim=-1)
    return probs[:, :-1].max(dim=1).values


def max_prob(_, output):
    probs = F.softmax(output.clamp(min=-25, max=25), dim=-1)
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
        self.model.train()
        preds = torch.max(output, dim=1).indices
        pred_probs = []
        for _ in range(self.n_forward_passes):
            dropout_output, _ = self.model(input, compute_conf=False)
            dropout_output = softmax(dropout_output)
            pred_probs.append(gold_values(dropout_output, preds))
        pred_probs = torch.stack(pred_probs)
        confs = torch.mean(pred_probs, dim=0)
        return confs


class Feedforward(nn.Module):

    def __init__(self,
                 input_size=784,
                 hidden_sizes=(128, 64),
                 output_size=10,
                 confidence_extractor='max_prob'):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.confidence_extractor = lookup_confidence_extractor(confidence_extractor, self)
        self.dropout = nn.Dropout(p=0.5)
        self.linears = nn.ModuleList([])
        self.linears.append(cudaify(nn.Linear(input_size, hidden_sizes[0])))
        for i in range(len(hidden_sizes)-1):
            self.linears.append(cudaify(nn.Linear(hidden_sizes[i], hidden_sizes[i+1])))
        self.final = cudaify(nn.Linear(hidden_sizes[-1], output_size))
        self.relu = nn.ReLU()

    def initial_layers(self, input_vec):
        nextout = cudaify(input_vec)
        for layer in self.linears:
            nextout = layer(nextout)
            nextout = self.relu(nextout)
            nextout = self.dropout(nextout)
        return nextout

    def final_layers(self, input_vec, orig_input_vec, compute_conf):
        nextout = self.final(input_vec)
        if compute_conf:
            confidences = self.confidence_extractor(orig_input_vec, nextout)
        else:
            confidences = None
        return nextout, confidences

    def forward(self, input_vec, compute_conf=True):
        nextout = self.initial_layers(input_vec)
        result, confidence = self.final_layers(nextout, input_vec, compute_conf)
        return result, confidence


class InterfaceAFeedforward(Feedforward):
    pass


class InterfaceBFeedforward(Feedforward):
 
    def __init__(self, 
                 input_size=784,
                 hidden_sizes=(128, 64),
                 output_size=10,
                 confidence_extractor='inv_abs'):
        super().__init__(input_size, hidden_sizes, output_size, confidence_extractor)
        self.final = cudaify(nn.Linear(hidden_sizes[1], output_size + 1))


class InterfaceCFeedforward(Feedforward):
 
    def __init__(self, 
                 input_size=784,
                 hidden_sizes=(128, 64),
                 output_size=10):
        super().__init__(input_size, hidden_sizes, output_size)
        self.confidence_layer = cudaify(nn.Linear(hidden_sizes[1], 1))

    def final_layers(self, input_vec, _, __):
        nextout = self.final(input_vec)
        confidence = self.confidence_layer(input_vec).reshape(-1)
        return nextout, confidence


class PretrainedTransformer(nn.Module):

    def __init__(self, confidence_extractor='max_prob'):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)
        self.confidence_extractor = lookup_confidence_extractor(confidence_extractor, self)

    def forward(self, batch, compute_conf=True):
        outputs = self.model(**batch).logits
        if compute_conf:
            confidence = self.confidence_extractor(batch['input_ids'], outputs)  # TODO: should we clone and detach?
        else:
            confidence = None
        return outputs, confidence
