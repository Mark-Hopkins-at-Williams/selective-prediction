from spred.viz import Visualizer
import matplotlib.pyplot as plt


def x_coords(pts):
    return [pt[0] for pt in pts]


def y_coords(pts):
    return [pt[1] for pt in pts]


class NormalsVisualizer(Visualizer):

    def __init__(self, last_epoch):
        super().__init__()
        self.last_epoch = last_epoch

    def visualize(self, epoch, val_loader, results):
        if epoch == self.last_epoch:
            val_instances, _ = next(iter(val_loader.restart()))
            pairs = [tuple(list(val_instances[i].numpy())[:2])
                     for i in range(len(val_instances))]
            class0 = [(pair, result) for (pair, result) in zip(pairs, results[:64])
                      if result['gold'] == 0]
            class1 = [(pair, result) for (pair, result) in zip(pairs, results[:64])
                      if result['gold'] == 1]
            pairs0 = [pair for (pair, _) in class0]
            confs0 = [result['confidence'] for (_, result) in class0]
            pairs1 = [pair for (pair, _) in class1]
            confs1 = [result['confidence'] for (_, result) in class1]
            plt.title('Confidence Visualization')
            plt.scatter(x_coords(pairs0), y_coords(pairs0), c=confs0,
                        label="class A", cmap='Reds')
            plt.scatter(x_coords(pairs1), y_coords(pairs1), c=confs1,
                        label="class B", cmap='Blues')
            plt.legend()
            plt.xlabel('x')
            plt.ylabel('y')
            plt.show()