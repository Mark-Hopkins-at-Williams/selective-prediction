import unittest
import torch
from spred.confidence import inv_abstain_prob, max_nonabstain_prob
from test.mnist_examples import build_interface_a_net, run_interface_a_input
from test.mnist_examples import build_interface_b_net, run_interface_b_input
from test.mnist_examples import build_interface_c_net


class TestMnistNetworks(unittest.TestCase):

    def test_basic_ffn(self):
        net = build_interface_a_net()
        conf = run_interface_a_input(net)
        expected_conf = torch.tensor([0.6188, 0.6646]).to('cpu')
        assert conf.shape == expected_conf.shape
        assert torch.allclose(conf, expected_conf, atol=10**(-4))

    def test_abstaining_ffn(self):
        net = build_interface_b_net(confidence=None)
        run_interface_b_input(net)
    
    def test_abstaining_ffn_with_inv_abstain_prob(self):
        net = build_interface_b_net(confidence=inv_abstain_prob)
        conf = run_interface_b_input(net).to('cpu')
        expected_conf = torch.tensor([0.6172, 0.5628]).to('cpu')
        assert conf.shape == expected_conf.shape
        assert torch.allclose(conf, expected_conf, atol=10**(-4))
        
    def test_abstaining_ffn_with_max_nonabstain_prob(self):
        net = build_interface_b_net(confidence=max_nonabstain_prob)
        conf = run_interface_b_input(net).to('cpu')
        expected_conf = torch.tensor([0.4551, 0.3366]).to('cpu')
        assert conf.shape == expected_conf.shape
        assert torch.allclose(conf, expected_conf, atol=10**(-4))

    def test_confident_ffn(self):
        net = build_interface_c_net()
        conf = run_interface_a_input(net)
        expected_conf = torch.tensor([1.6482, 2.1929]).to('cpu')
        assert conf.shape == expected_conf.shape
        assert torch.allclose(conf, expected_conf, atol=10**(-4))


if __name__ == "__main__":
    unittest.main()
