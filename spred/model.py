from torch import nn
from transformers import AutoModelForSequenceClassification
from spred.hub import spred_hub
from spred.loss import init_loss_fn

def init_model(model_config, regularizer, include_abstain):
    architecture = model_config['name']
    params = {k: model_config[k] for k in model_config
              if k not in ['name', 'loss']}
    if 'loss' in model_config:
        params['loss_f'] = init_loss_fn(model_config['loss'])
    params['incl_abstain'] = include_abstain
    model_constructor = spred_hub.get_model(architecture)
    base_model = model_constructor(**params)
    if regularizer is not None:
        return RegularizedModel(base_model, regularizer, include_abstain)
    else:
        return base_model


class SelectiveModel(nn.Module):

    def __init__(self, incl_abstain):
        super().__init__()
        self.epoch = 0
        self.confidence_extractor = None
        self.incl_abstain = incl_abstain

    def set_confidence_extractor(self, extractor):
        self.confidence_extractor = extractor

    def notify(self, epoch):
        self.epoch = epoch

    def include_abstain(self):
        return self.incl_abstain



class Feedforward(SelectiveModel):

    def __init__(self, input_size, hidden_sizes, output_size,
                 loss_f, incl_abstain):
        super().__init__(incl_abstain)
        self.input_size = input_size
        self.output_size = output_size + 1 if self.include_abstain() else output_size
        self.confidence_extractor = None
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

    def __init__(self, base_model, output_size, incl_abstain):
        super().__init__(incl_abstain)
        self.output_size = output_size + 1 if self.include_abstain() else output_size
        self.model = AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=self.output_size)
        self.confidence_extractor = None

    def lite_forward(self, batch):
        """ For use by MC Dropout. """
        self.model.train()
        outputs = self.model(**batch)
        return {'outputs': outputs.logits.detach(), 'loss': outputs.loss.detach()}

    def embed(self, batch):
        """ For use by Trustscore. """
        self.model.train()
        try: # TODO: fix this hack
            b = batch['inputs']
        except:
            b = batch
        outputs = self.model(**b, output_hidden_states=True)
        t = outputs.hidden_states[-1]
        return {'outputs': t[:,0,:].detach()}

    def forward(self, batch, compute_conf=True, compute_loss=True):
        outputs = self.model(**batch)
        if compute_conf:
            print(outputs.logits)
            confidence = self.confidence_extractor({'inputs': batch,
                                                    'outputs': outputs.logits}, self.lite_forward)
        else:
            confidence = None
        if compute_loss:
            return {'outputs': outputs.logits, 'loss': outputs.loss,
                    'confidences': confidence}
        else:
            return {'outputs': outputs.logits, 'confidences': confidence}


class RegularizedModel(SelectiveModel):

    def __init__(self, base_model, regularizer, incl_abstain):
        super().__init__(incl_abstain)
        self.base_model = base_model
        self.regularizer = regularizer

    def forward(self, batch, compute_conf=True):
        model_out = self.base_model.forward(batch, compute_conf, compute_loss=True)
        model_out['labels'] = batch['labels']
        regularized_loss = self.regularizer(model_out)
        model_out['loss'] = regularized_loss
        return model_out

    def notify(self, epoch):
        self.base_model.notify(epoch)
        self.regularizer.notify(epoch)

    def set_confidence_extractor(self, extractor):
        self.base_model.confidence_extractor = extractor


spred_hub.register_model("feedforward", Feedforward)
spred_hub.register_model("pretrained", PretrainedTransformer)
