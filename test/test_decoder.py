import unittest
import torch
from torch import tensor
from spred.model import InterfaceAFeedforward
from spred.model import InterfaceBFeedforward
from spred.confidence import max_nonabstain_prob
from spred.decoder import InterfaceADecoder, InterfaceBDecoder, InterfaceCDecoder
from spred.util import close_enough
from test.model_examples import build_interface_a_net, build_interface_b_net
from test.model_examples import ExampleFeedforwardLoader


class TestDecoder(unittest.TestCase):

    def test_interface_a_decoder(self):
        net = build_interface_a_net()
        decoder = InterfaceADecoder()
        loader = ExampleFeedforwardLoader()
        decoded = decoder(net, loader)
        result = []
        for x in decoded:
            x['confidence'] = round(x['confidence'], 4)
            result.append(x)
        expected = [{'pred': 1, 'gold': 1, 'confidence': 1.0, 'abstain': False},
                    {'pred': 1, 'gold': 0, 'confidence': 0.9975, 'abstain': False}]
        assert result == expected

    """
    def test_interface_b_mnist(self):
        torch.manual_seed(1977)
        net = build_interface_b_net(max_nonabstain_prob)
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
    """


if __name__ == "__main__":
    unittest.main()
