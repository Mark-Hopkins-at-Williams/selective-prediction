import torch
from torch import nn
import torch.nn.functional as F
from spred.util import cudaify


def inv_abstain_prob(output_tensor):
    probs = F.softmax(output_tensor.clamp(min=-25, max=25), dim=-1)
    return 1.0 - probs[:,-1]


def max_nonabstain_prob(output_tensor):
    probs = F.softmax(output_tensor.clamp(min=-25, max=25), dim=-1)
    return probs[:, :-1].max(dim=1).values


def max_prob(output_tensor):
    probs = F.softmax(output_tensor.clamp(min=-25, max=25), dim=-1)
    return probs.max(dim=1).values


def abstention(output_tensor):
    return output_tensor[:, -1]


def random_confidence(output_tensor):
    return torch.randn(output_tensor.shape[0])


confidence_extractor_lookup = {'inv_abs': inv_abstain_prob,
                               'max_non_abs': max_nonabstain_prob,
                               'abs': abstention,
                               'max_prob': max_prob,
                               'random': random_confidence}


class Feedforward(nn.Module):

    def __init__(self,
                 input_size=784,
                 hidden_sizes=(128, 64),
                 output_size=10,
                 confidence_extractor='max_prob'):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.confidence_extractor = confidence_extractor_lookup[confidence_extractor]
        self.dropout = nn.Dropout(p=0.2)
        self.linear1 = cudaify(nn.Linear(input_size, hidden_sizes[0]))
        self.linear2 = cudaify(nn.Linear(hidden_sizes[0], hidden_sizes[1]))
        self.final = cudaify(nn.Linear(hidden_sizes[1], output_size))
        self.softmax = cudaify(nn.Softmax(dim=1))
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()

    def initial_layers(self, input_vec):
        nextout = cudaify(input_vec)
        nextout = self.linear1(nextout)
        nextout = self.relu1(nextout)
        nextout = self.dropout(nextout)
        nextout = self.linear2(nextout)
        nextout = self.relu2(nextout)
        nextout = self.dropout(nextout)
        return nextout

    def final_layers(self, input_vec):
        nextout = self.final(input_vec)
        confidences = self.confidence_extractor(nextout)
        return nextout, confidences

    def forward(self, input_vec):
        nextout = self.initial_layers(input_vec)
        result, confidence = self.final_layers(nextout)
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

    def final_layers(self, input_vec):
        nextout = self.final(input_vec)
        confidence = self.confidence_layer(input_vec).reshape(-1)
        return nextout, confidence

