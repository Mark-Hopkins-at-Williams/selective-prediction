import unittest
import torch
import os
from os.path import join
from torchvision import datasets, transforms
import sys
file_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(1, join(file_dir, ".."))
from spred.tasks.mnist import confuse_two, MnistLoader, MnistPairLoader


def visualize_mnist_image(image_tensor):
    lines = []
    for row in image_tensor:
        line = ''.join(['X' if x > 0 else '.' for x in row])
        lines.append(line)
    return '\n'.join(lines)


def tensor_eq(t1, t2):
    return torch.all(t1.eq(t2))


def load_mnist_data():
    data_dir = join(file_dir, "data")
    train_dir = join(data_dir, "train")
    trainset = datasets.MNIST(train_dir,
                              download=True,
                              train=True,
                              transform=transforms.ToTensor())
    return trainset


IMAGE0 = ['............................',
          '............................',
          '............................',
          '............................',
          '............................',
          '............XXXXXXXXXXXX....',
          '........XXXXXXXXXXXXXXXX....',
          '.......XXXXXXXXXXXXXXXX.....',
          '.......XXXXXXXXXXX..........',
          '........XXXXXXX.XX..........',
          '.........XXXXX..............',
          '...........XXXX.............',
          '...........XXXX.............',
          '............XXXXXX..........',
          '.............XXXXXX.........',
          '..............XXXXXX........',
          '...............XXXXX........',
          '.................XXXX.......',
          '..............XXXXXXX.......',
          '............XXXXXXXX........',
          '..........XXXXXXXXX.........',
          '........XXXXXXXXXX..........',
          '......XXXXXXXXXX............',
          '....XXXXXXXXXX..............',
          '....XXXXXXXX................',
          '............................',
          '............................',
          '............................']

IMAGE1 = ['............................',
          '............................',
          '............................',
          '............................',
          '...............XXXXX........',
          '..............XXXXXX........',
          '.............XXXXXXXXX......',
          '...........XXXXXXXXXXX......',
          '...........XXXXXXXXXXX......',
          '..........XXXXXXXXXXXX......',
          '.........XXXXXXXXX..XXX.....',
          '........XXXXXX......XXX.....',
          '.......XXXXXXX......XXX.....',
          '.......XXXX.........XXX.....',
          '.......XXX..........XXX.....',
          '......XXXX..........XXX.....',
          '......XXXX........XXXXX.....',
          '......XXX........XXXXX......',
          '......XXX.......XXXX........',
          '......XXX......XXXX.........',
          '......XXXXXXXXXXXXX.........',
          '......XXXXXXXXXXX...........',
          '......XXXXXXXXX.............',
          '.......XXXXXXX..............',
          '............................',
          '............................',
          '............................',
          '............................']


class TestMnistLoader(unittest.TestCase):

    def test_confuse(self):
        torch.manual_seed(1234567)
        labels = torch.tensor([4, 2, 9, 0, 1, 8, 0, 1, 7, 7])
        new_labels1 = confuse_two(labels)
        new_labels2 = confuse_two(labels)
        new_labels3 = confuse_two(labels)
        assert(new_labels1.equal(torch.tensor([4, 2, 9, 0, 1, 8, 0, 1, 7, 7])))
        assert(new_labels2.equal(torch.tensor([4, 2, 9, 0, 1, 8, 0, 7, 7, 1])))
        assert(new_labels3.equal(torch.tensor([4, 2, 9, 0, 7, 8, 0, 1, 7, 7])))

    def test_loader(self):
        trainset = load_mnist_data()
        loader = MnistLoader(trainset, bsz=2, shuffle=False)
        assert len(loader) == 30000
        for batch_x, batch_y in loader:
            loaded0 = batch_x[0].reshape(28,28)
            assert visualize_mnist_image(loaded0) == '\n'.join(IMAGE0)
            assert tensor_eq(batch_y, torch.tensor([5, 0]))
            loaded1 = batch_x[1].reshape(28,28)
            assert visualize_mnist_image(loaded1) == '\n'.join(IMAGE1)
            break

    def test_pair_loader(self):
        trainset = load_mnist_data()
        loader = MnistPairLoader(trainset, bsz=2, shuffle=False)
        assert len(loader) == 30000
        for batch_x_1, batch_x_2, batch_y_1, batch_y_2 in loader:
            loaded10 = batch_x_1[0].reshape(28, 28)
            assert visualize_mnist_image(loaded10) == '\n'.join(IMAGE0)
            loaded11 = batch_x_1[1].reshape(28, 28)
            assert visualize_mnist_image(loaded11) == '\n'.join(IMAGE1)
            loaded20 = batch_x_2[0].reshape(28, 28)
            assert visualize_mnist_image(loaded20) == '\n'.join(IMAGE0)
            loaded21 = batch_x_2[1].reshape(28, 28)
            assert visualize_mnist_image(loaded21) == '\n'.join(IMAGE1)
            break


if __name__ == '__main__':
    unittest.main()
