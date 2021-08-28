from abc import ABC, abstractmethod


class Visualizer(ABC):

    @abstractmethod
    def visualize(self, epoch, val_loader, results):
        pass
