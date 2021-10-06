import unittest
from spred.decoder import Decoder
from test.examples import build_interface_a_net, build_interface_b_net
from test.examples import ExampleFeedforwardLoader


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
        expected = [{'pred': 1, 'gold': 1, 'confidence': 0.5587, 'non_abstain_prob': 1.0},
                    {'pred': 1, 'gold': 0, 'confidence': 0.5866, 'non_abstain_prob': 1.0}]
        assert result == expected

    def test_interface_b_decoder(self):
        net = build_interface_b_net()
        decoder = Decoder(include_abstain_output=True)
        loader = ExampleFeedforwardLoader()
        decoded = decoder(net, loader)
        result = []
        for x in decoded:
            x['confidence'] = round(x['confidence'], 4)
            x['non_abstain_prob'] = round(x['non_abstain_prob'], 4)
            result.append(x)
        expected = [{'pred': 1, 'gold': 1, 'confidence': 0.3313, 'non_abstain_prob': 0.6204},
                    {'pred': 1, 'gold': 0, 'confidence': 0.3278, 'non_abstain_prob': 0.5849}]
        assert result == expected


if __name__ == "__main__":
    unittest.main()
