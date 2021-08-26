import torch
from spred.mnist.model import InterfaceAFeedforward
from spred.mnist.model import InterfaceBFeedforward
from spred.mnist.model import InterfaceCFeedforward


class ExampleMnistLoader:

    def __init__(self):
        self.images = torch.tensor([[-2., 1.], [5., 2.]])
        self.labels = torch.tensor([3, 5])

    def __iter__(self):
        yield self.images, self.labels

    def __len__(self):
        return 1


def set_ffn_params(net):
    for param in net.parameters():
        if param.shape == torch.Size([3]):
            param[0] = 1.4640
            param[1] = -0.3238
            param[2] = 0.7740
        elif param.shape == torch.Size([2, 2]):
            param[0][0] = 0.1940
            param[0][1] = 2.1614
            param[1][0] = -0.1721
            param[1][1] = -0.1721
        elif param.shape == torch.Size([2]):
            param[0] = 0.1391
            param[1] = -0.1082
        elif param.shape == torch.Size([3, 2]):
            param[0][0] = -1.2682
            param[0][1] = -0.0383
            param[1][0] = -0.1029
            param[1][1] = 1.4400
            param[2][0] = -0.4705
            param[2][1] = 1.1624
        else:
            torch.nn.init.ones_(param)


def build_interface_a_net():
    net = InterfaceAFeedforward(input_size=2, hidden_sizes=[2, 2], output_size=2)
    set_ffn_params(net)
    net.eval()
    return net


def run_interface_a_input(net):
    result, conf = net(torch.tensor([[-2., 1.], [5., 2.]]))
    result = torch.softmax(result.to('cpu'), dim=1)
    expected = torch.tensor([[0.6188, 0.3812],
                             [0.6646, 0.3354]]).to('cpu')
    assert result.shape == expected.shape
    assert torch.allclose(result, expected, atol=10**(-4))
    return conf


def build_interface_b_net(confidence):
    net = InterfaceBFeedforward(input_size=2, hidden_sizes=[2, 2], output_size=2)
    set_ffn_params(net)
    if confidence is not None:
        net.confidence_extractor = confidence
    net.eval()
    return net


def run_interface_b_input(net):
    result, conf = net(torch.tensor([[-2., 1.], [5., 2.]]))
    result = torch.softmax(result.to('cpu'), dim=1)
    expected = torch.tensor([[0.4551, 0.1621, 0.3828],
                             [0.3366, 0.2261, 0.4372]]).to('cpu')
    assert result.shape == expected.shape
    assert torch.allclose(result, expected, atol=10 ** (-4))
    return conf


def build_interface_c_net():
    net = InterfaceCFeedforward(input_size=2, hidden_sizes=[2, 2], output_size=2)
    set_ffn_params(net)
    net.eval()
    return net
