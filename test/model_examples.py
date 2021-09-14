import torch
from torch import tensor
from spred.model import Feedforward
from spred.loss import CrossEntropyLoss
from spred.confidence import max_nonabstain_prob, max_prob



class ExampleFeedforwardLoader:

    def __init__(self):
        self.images = torch.tensor([[-2., 1.], [5., 2.]])
        self.labels = torch.tensor([1, 0])

    def __iter__(self):
        yield {'inputs': self.images, 'labels': self.labels}

    def __len__(self):
        return 1


def set_ffn_params(net):
    for param in net.parameters():
        if param.shape == torch.Size([3]):
            param.data = tensor([1.0, 2.0, 3.0])
        elif param.shape == torch.Size([2, 2]):
            param.data = tensor([[-2.0, 1.0],
                                 [2.0, 3.0]])
        elif param.shape == torch.Size([2]):
            param.data = tensor([-1.0, 1.0])
        elif param.shape == torch.Size([3, 2]):
            param.data = tensor([[-1.0, 1.0],
                                 [2.0, 3.0],
                                 [4.0, 5.0]])
        else:
            torch.nn.init.ones_(param)


def build_interface_a_net():
    net = Feedforward(input_size=2, hidden_sizes=[2, 2], output_size=2,
                      loss_f=CrossEntropyLoss(),
                      confidence_extractor=max_prob,
                      include_abstain_output=False)
    set_ffn_params(net)
    net.eval()
    return net


def build_interface_b_net():
    net = Feedforward(input_size=2, hidden_sizes=[2, 2], output_size=2,
                      loss_f=CrossEntropyLoss(),
                      confidence_extractor=max_nonabstain_prob,
                      include_abstain_output=True)
    set_ffn_params(net)
    net.eval()
    return net
