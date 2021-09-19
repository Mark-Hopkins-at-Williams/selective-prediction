from torch import nn
from transformers import AutoModelForSequenceClassification

class SelectiveModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.epoch = 0
        self.confidence_extractor = None

    def set_confidence_extractor(self, extractor):
        self.confidence_extractor = extractor

    def notify(self, epoch):
        self.epoch = epoch


class Feedforward(SelectiveModel):

    def __init__(self, input_size, hidden_sizes, output_size,
                 loss_f, confidence_extractor, include_abstain_output=False):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size + 1 if include_abstain_output else output_size
        self.confidence_extractor = confidence_extractor
        self.dropout = nn.Dropout(p=0.5)
        self.linears = nn.ModuleList([])
        self.linears.append(nn.Linear(input_size, hidden_sizes[0]))
        for i in range(len(hidden_sizes)-1):
            self.linears.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
        self.final = nn.Linear(hidden_sizes[-1], self.output_size)
        self.relu = nn.ReLU()
        self.loss_f = loss_f

    def notify(self, epoch):
        self.loss_f.notify(epoch)

    def initial_layers(self, input_vec):
        nextout = input_vec
        for layer in self.linears:
            nextout = layer(nextout)
            nextout = self.relu(nextout)
            nextout = self.dropout(nextout)
        return nextout

    def final_layers(self, input_vec, orig_input_vec, compute_conf):
        nextout = self.final(input_vec)
        if compute_conf:
            confidences = self.confidence_extractor({'inputs': orig_input_vec['inputs'],
                                                     'outputs': nextout}, self.lite_forward)
        else:
            confidences = None
        return nextout, confidences

    def lite_forward(self, batch):
        """ For use by MC Dropout. """
        self.train()
        nextout = self.initial_layers(batch)
        nextout = self.final(nextout)
        return {'outputs': nextout.detach()}

    def embed(self, batch):
        """ For use by Trustscore. """
        self.train()
        nextout = self.initial_layers(batch['inputs'])
        return {'outputs': nextout.detach()}

    def forward(self, batch, compute_conf=True, compute_loss=True):
        nextout = self.initial_layers(batch['inputs'])
        result, confidence = self.final_layers(nextout, batch, compute_conf)
        if compute_loss:
            loss = self.loss_f({'outputs': result,
                                'confidences': confidence,
                                'labels': batch['labels']})
        else:
            loss = None
        return {'outputs': result, 'loss': loss, 'confidences': confidence}



class PretrainedTransformer(SelectiveModel):

    def __init__(self, base_model, confidence_extractor, output_size,
                 include_abstain_output=False):
        super().__init__()
        self.output_size = output_size + 1 if include_abstain_output else output_size
        self.model = AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=self.output_size)
        self.confidence_extractor = confidence_extractor

    def lite_forward(self, batch):
        """ For use by MC Dropout. """
        self.model.train()
        outputs = self.model(**batch)
        return {'outputs': outputs.logits.detach(), 'loss': outputs.loss.detach()}

    def embed(self, batch):
        """ For use by Trustscore. """
        self.model.train()
        outputs = self.model(**batch, output_hidden_states=True)
        t = outputs.hidden_states[-1]
        return {'outputs': t[:,0,:].detach()}

    def forward(self, batch, compute_conf=True, compute_loss=True):
        outputs = self.model(**batch)
        if compute_conf:
            confidence = self.confidence_extractor({'inputs': batch,
                                                    'outputs': outputs.logits}, self.lite_forward)  # TODO: should we clone and detach?
        else:
            confidence = None
        if compute_loss:
            return {'outputs': outputs.logits, 'loss': outputs.loss,
                    'confidences': confidence}
        else:
            return {'outputs': outputs.logits, 'confidences': confidence}

