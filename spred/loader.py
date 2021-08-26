class Loader:
    def __iter__(self):
        raise NotImplementedError("This feature needs to be implemented in the child class.")

    def __len__(self):
        raise NotImplementedError("This feature needs to be implemented in the child class.")
