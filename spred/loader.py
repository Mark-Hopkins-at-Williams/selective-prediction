from abc import ABC, abstractmethod


class Loader(ABC):
    @abstractmethod
    def __iter__(self):
        ...

    @abstractmethod
    def __len__(self):
        ...

    @abstractmethod
    def input_size(self):
        ...

    @abstractmethod
    def output_size(self):
        ...
