from abc import ABC, abstractmethod


class Loader(ABC):
    @abstractmethod
    def __iter__(self):
        ...

    @abstractmethod
    def __len__(self):
        ...
