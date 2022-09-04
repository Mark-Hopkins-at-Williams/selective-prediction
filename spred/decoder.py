"""
Decoder.py includes the Inferential functionalities of the package. The ```Decoder``` class
configures the selective prediction process and is able to return relevant model outputs upon
request

"""

import torch
from torch.nn import functional
from tqdm import tqdm
from datasets import load_metric
from spred.evaluate import Evaluator
from spred.util import softmax


def validate_and_analyze(model, validation_loader, task_name=None):
    """
    This function takes a model, and validation dataloader, and configures the
    conrresponding ```Evaluator``` to the task.

    """
    decoder = Decoder(model.include_abstain())
    model.eval()
    results = list(decoder(model, validation_loader))
    validation_loss = decoder.get_loss()
    eval_result = Evaluator(results, validation_loss, task_name).get_result()
    return eval_result


class Decoder:
    """
    Decoder implements the relevent functionalities needed for selective inference.
    See the comment on each method for details.

    """

    def __init__(self, include_abstain_output):
        """
        Configures whether the model runs on GPU or CPU and whether the model has
        an abstention output

        """
        self.device = (torch.device("cuda") if torch.cuda.is_available()
                       else torch.device("cpu"))
        self.include_abstain_output = include_abstain_output

    def get_loss(self):
        """
        Decoder keeps track of the running loss during validation
        This function returns the average running loss

        """
        if self.running_loss_denom == 0:
            return None
        return self.running_loss_total / self.running_loss_denom

    def make_predictions(self, outputs, labels, conf):
        """
        This function returns a dictionary of features extracted from a selective predictor, which includes
        - ```pred```: the index corresponding to the predicted class
        - ```gold```: the ground truth
        - ```confidence```: the confidence scores
        - ```non_abstain_prob```: this field has value 1 unless ```non_abstain_prob``` is ```True```.
                                  ```non_abstain_prob``` should only be set to `True` if the you are using
                                  the 1's complement of the abstention output as confidence. In that case,
                                  use this field as the confidence score.
        
        """
        if not self.include_abstain_output:
            preds = outputs.argmax(dim=1)
            for p, g, c in zip(preds, labels, conf):
                yield {'pred': p.item(), 'gold': g.item(),
                       'confidence': c.item(), 'non_abstain_prob': 1.0}
        else:
            output = softmax(outputs)
            abstain_probs = output[:, -1]
            preds = output.argmax(dim=-1)
            abs_i = output.shape[1] - 1
            preds[preds == abs_i] = -1
            no_abstain_preds = output[:, :-1].argmax(dim=-1)
            for element in zip(no_abstain_preds, labels, conf, abstain_probs):
                p, g, c, p2 = element
                result = {'pred': p.item(), 'gold': g.item(),
                          'confidence': c.item(), 'non_abstain_prob': (1.0-p2).item()}
                yield result

    def __call__(self, model, data, loss_f=None):
        """
        Take a model and dataloader, return the ```make_prediction``` dictionary
        from the model output on the loader examples

        """
        self.running_loss_total = 0.0
        self.running_loss_denom = 0
        for batch in tqdm(data, total=len(data)):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            model.eval()
            with torch.no_grad():
                model_out = model(batch, compute_conf=True)
                outputs, loss, conf = model_out['outputs'], model_out['loss'], model_out['confidences']
            self.running_loss_total += loss.item()
            self.running_loss_denom += 1
            for pred in self.make_predictions(outputs, batch['labels'], conf):
                yield pred

