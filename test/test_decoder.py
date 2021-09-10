import unittest
import torch
from spred.model import InterfaceAFeedforward
from spred.model import InterfaceBFeedforward
from spred.confidence import max_nonabstain_prob
from spred.tasks.mnist import MnistLoader
from spred.decoder import InterfaceADecoder, InterfaceBDecoder, InterfaceCDecoder
from test.mnist.test_mnist_loader import load_mnist_data
from spred.tasks.mnist.examples import build_interface_a_net
from spred.tasks.mnist.examples import build_interface_b_net
from spred.tasks.mnist.examples import build_interface_c_net
from spred.tasks.mnist.examples import ExampleMnistLoader


class TestMnistDecoder(unittest.TestCase):

    def test_interface_a_mnist(self):
        torch.manual_seed(1977)
        net = InterfaceAFeedforward()
        net.eval()
        decoder = InterfaceADecoder()
        trainset = load_mnist_data()
        loader = MnistLoader(trainset, bsz=2, shuffle=False)
        decoded = decoder(net, loader)
        result = []
        for i, x in enumerate(decoded):
            x['confidence'] = round(x['confidence'], 4)
            result.append(x)
            if i == 4:
                break
        expected = [{'pred': 6, 'gold': 5, 'confidence': 0.1132, 'abstain': False},
                    {'pred': 7, 'gold': 0, 'confidence': 0.1128, 'abstain': False},
                    {'pred': 7, 'gold': 4, 'confidence': 0.1147, 'abstain': False},
                    {'pred': 8, 'gold': 1, 'confidence': 0.1125, 'abstain': False},
                    {'pred': 7, 'gold': 9, 'confidence': 0.1109, 'abstain': False}]
        assert result == expected

    def test_interface_b_mnist(self):
        torch.manual_seed(1977)
        net = InterfaceBFeedforward(confidence_extractor='inv_abs')
        net.eval()
        decoder = InterfaceBDecoder()
        trainset = load_mnist_data()
        loader = MnistLoader(trainset, bsz=2, shuffle=False)
        decoded = decoder(net, loader)
        result = []
        for i, x in enumerate(decoded):
            x['confidence'] = round(x['confidence'], 4)
            result.append(x)
            if i > 3:
                break
        expected = [{'pred': 1, 'gold': 5, 'confidence': 0.8992, 'abstain': True},
                    {'pred': 1, 'gold': 0, 'confidence': 0.8977, 'abstain': True},
                    {'pred': 1, 'gold': 4, 'confidence': 0.8982, 'abstain': True},
                    {'pred': 8, 'gold': 1, 'confidence': 0.9006, 'abstain': True},
                    {'pred': 1, 'gold': 9, 'confidence': 0.9019, 'abstain': False}]
        assert result == expected

    def test_interface_a_decoder(self):
        torch.manual_seed(1977)
        net = build_interface_a_net()
        decoder = InterfaceADecoder()
        loader = ExampleMnistLoader()
        decoded = decoder(net, loader)
        result = []
        for i, x in enumerate(decoded):
            x['confidence'] = round(x['confidence'], 4)
            result.append(x)
        expected = [{'pred': 0, 'gold': 3, 'confidence': 0.6188, 'abstain': False},
                    {'pred': 0, 'gold': 5, 'confidence': 0.6646, 'abstain': False}]
        assert result == expected

    def test_interface_b_decoder(self):
        torch.manual_seed(1977)
        net = build_interface_b_net(max_nonabstain_prob)
        decoder = InterfaceBDecoder()
        loader = ExampleMnistLoader()
        decoded = decoder(net, loader)
        result = []
        for i, x in enumerate(decoded):
            x['confidence'] = round(x['confidence'], 4)
            result.append(x)
        expected = [{'pred': 0, 'gold': 3, 'confidence': 0.4551, 'abstain': False},
                    {'pred': 0, 'gold': 5, 'confidence': 0.3366, 'abstain': True}]
        assert result == expected

    def test_interface_c_decoder(self):
        torch.manual_seed(1977)
        net = build_interface_c_net()
        decoder = InterfaceCDecoder()
        loader = ExampleMnistLoader()
        decoded = decoder(net, loader)
        result = []
        for i, x in enumerate(decoded):
            x['confidence'] = round(x['confidence'], 4)
            result.append(x)
        expected = [{'pred': 0, 'gold': 3, 'confidence': 1.6482, 'abstain': False},
                    {'pred': 0, 'gold': 5, 'confidence': 2.1929, 'abstain': False}]
        assert result == expected


if __name__ == "__main__":
    unittest.main()
