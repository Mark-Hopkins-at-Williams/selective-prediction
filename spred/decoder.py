import torch
from torch.nn import functional
from tqdm import tqdm
from datasets import load_metric
from spred.analytics import Evaluator


def validate_and_analyze(model, validation_loader, epoch=0, visualizer=None):
    decoder = Decoder()
    model.eval()
    results = list(decoder(model, validation_loader))
    validation_loss = decoder.get_loss()
    if visualizer is not None:
        visualizer.visualize(epoch, validation_loader, results)
    eval_result = Evaluator(results, validation_loss).get_result()
    return eval_result


class Decoder:

    def __init__(self, include_abstain_output=False):
        self.device = (torch.device("cuda") if torch.cuda.is_available()
                       else torch.device("cpu"))
        self.include_abstain_output = include_abstain_output

    def get_loss(self):
        if self.running_loss_denom == 0:
            return None
        return self.running_loss_total / self.running_loss_denom

    def make_predictions(self, outputs, labels, conf):
        if not self.include_abstain_output:
            preds = outputs.argmax(dim=1)
            for p, g, c in zip(preds, labels, conf):
                yield {'pred': p.item(), 'gold': g.item(),
                       'confidence': c.item(), 'abstain': False}
        else:
            output = functional.softmax(outputs.clamp(min=-25, max=25), dim=1)
            preds = output.argmax(dim=-1)
            abs_i = output.shape[1] - 1
            preds[preds == abs_i] = -1
            no_abstain_preds = output[:, :-1].argmax(dim=-1)
            for element in zip(no_abstain_preds, labels, conf, preds):
                p, g, c, p2 = element
                result = {'pred': p.item(), 'gold': g.item(),
                          'confidence': c.item(), 'abstain': p2.item() == -1}
                yield result

    def __call__(self, model, data, loss_f=None):
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

