import unittest
import torch
from torch import tensor
from spred.model import Feedforward
from spred.confidence import max_nonabstain_prob
from spred.decoder import Decoder
from spred.util import close_enough
from test.model_examples import build_interface_a_net, build_interface_b_net
from test.model_examples import ExampleFeedforwardLoader


class TestDecoder(unittest.TestCase):

    def test_interface_a_decoder(self):
        net = build_interface_a_net()
        decoder = Decoder()
        loader = ExampleFeedforwardLoader()
        decoded = decoder(net, loader)
        result = []
        for x in decoded:
            x['confidence'] = round(x['confidence'], 4)
            result.append(x)
        expected = [{'pred': 1, 'gold': 1, 'confidence': 1.0, 'abstain': False},
                    {'pred': 1, 'gold': 0, 'confidence': 0.9975, 'abstain': False}]
        assert result == expected


if __name__ == "__main__":
    unittest.main()
