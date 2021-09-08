Selective Prediction
--------------------

## When running PyTorch on Zebra, create a virtual env and install torch from source:

    pip3 install --pre torch torchvision torchaudio -f https://download.pytorch.org/whl/nightly/cu111/torch_nightly.html

## Installing the spred package:
From the top-level directory:

    pip install -e .


## Running Experiments
Run a basic experiment on MNIST: 
    
    python main.py

Some other example configuration files are found in the `config` directory.


## Testing
### To run all unit tests

From the top-level directory, run: 

    python -m unittest

### To run a particular unit test module (e.g. test/test_analytics.py)

From the top-level directory, run:

    python -m unittest test.test_analytics
