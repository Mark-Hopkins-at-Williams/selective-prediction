spred (a python package for selective prediction)
-------------------------------------------------

## Installing the spred package:

Create a clean Python environment and then from the top-level directory of
this repository, type the following:

    pip install -e .

If you have issues with the above command, it may help to refresh your pip:

    python -m pip install -U --force-reinstall pip    


## Running Experiments

You can validate your installation by running a basic experiment on MNIST: 
    
    python main.py -c config/small.exp.config.json tmp

Some other example experiment configuration files are found in 
the `config` directory.

https://reed-edu.zoom.us/j/93658144530
## Testing
### To run all unit tests

From the top-level directory, run: 

    python -m unittest

### To run a particular unit test module (e.g. test/test_analytics.py)

From the top-level directory, run:

    python -m unittest test.test_analytics


## Using and extending the package

```spred``` is intended to be used to evaluate existing
selective prediction techniques on a novel task, or to evaluate
novel selective prediction techniques on existing tasks. To learn how
to use and extend the package, we have provided several tutorials 
(in the form of Jupyter notebooks):

- ```docs/task_tutorial.ipynb```: New ```spred``` users should begin 
with this tutorial, which describes how to add a new task for evaluating
selective prediction techniques.
- ```docs/confidence_tutorial.ipynb```: This tutorial describes how to
add a new confidence function to ```spred```.
- ```docs/loss_tutorial.ipynb```: This tutorial describes how to add a
new loss function to ```spred```.

