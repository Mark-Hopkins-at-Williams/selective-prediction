import torch


def approx(x, y, num_sig=3):
    return abs(x-y) < 1.0 * (10 ** (-num_sig))


class Logger:
    def __init__(self, verbose):
        self.verbose = verbose
    
    def __call__(self, msg):
        if self.verbose:
            print(msg)


def cudaify(x):
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 2:
            cuda = torch.device('cuda:2')
        else:
            cuda = torch.device('cuda:0')
        return x.to(cuda)
    else: 
        return x


def predict_simple(output):
    return output.argmax(dim=1)


def predict_abs(output):
    return output[:, :-1].argmax(dim=1)
