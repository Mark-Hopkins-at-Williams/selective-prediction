from torch import nn
from spred.util import cudaify
from spred.confidence import lookup_confidence_extractor
from transformers import AutoModelForSequenceClassification


class Feedforward(nn.Module):

    def __init__(self, input_size, hidden_sizes, output_size,
                 loss_f, confidence_extractor='max_prob'):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.confidence_extractor = lookup_confidence_extractor(confidence_extractor, self)
        self.dropout = nn.Dropout(p=0.5)
        self.linears = nn.ModuleList([])
        self.linears.append(cudaify(nn.Linear(input_size, hidden_sizes[0])))
        for i in range(len(hidden_sizes)-1):
            self.linears.append(cudaify(nn.Linear(hidden_sizes[i], hidden_sizes[i+1])))
        self.final = cudaify(nn.Linear(hidden_sizes[-1], output_size))
        self.relu = nn.ReLU()
        self.loss_f = loss_f

    def initial_layers(self, input_vec):
        nextout = cudaify(input_vec)
        for layer in self.linears:
            nextout = layer(nextout)
            nextout = self.relu(nextout)
            nextout = self.dropout(nextout)
        return nextout

    def final_layers(self, input_vec, orig_input_vec, compute_conf):
        nextout = self.final(input_vec)
        if compute_conf:
            confidences = self.confidence_extractor(orig_input_vec, nextout)
        else:
            confidences = None
        return nextout, confidences

    def forward(self, batch, compute_conf=True):
        nextout = self.initial_layers(batch['input_ids'])
        result, confidence = self.final_layers(nextout, batch, compute_conf)
        loss = self.loss_f(result, confidence, batch['labels'])
        return result, loss, confidence


class InterfaceAFeedforward(Feedforward):
    pass


class InterfaceBFeedforward(Feedforward):
 
    def __init__(self, input_size, hidden_sizes, output_size,
                 confidence_extractor='inv_abs'):
        super().__init__(input_size, hidden_sizes, output_size, confidence_extractor)
        self.final = cudaify(nn.Linear(hidden_sizes[-1], output_size + 1))


class InterfaceCFeedforward(Feedforward):
 
    def __init__(self, 
                 input_size=784,
                 hidden_sizes=(128, 64),
                 output_size=10):
        super().__init__(input_size, hidden_sizes, output_size)
        self.confidence_layer = cudaify(nn.Linear(hidden_sizes[1], 1))

    def final_layers(self, input_vec, _, __):
        nextout = self.final(input_vec)
        confidence = self.confidence_layer(input_vec).reshape(-1)
        return nextout, confidence


class PretrainedTransformer(nn.Module):

    def __init__(self, base_model, confidence_extractor='max_prob'):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=2)
        self.confidence_extractor = lookup_confidence_extractor(confidence_extractor, self)

    def forward(self, batch, compute_conf=True):
        outputs = self.model(**batch)
        if compute_conf:
            confidence = self.confidence_extractor(batch, outputs.logits)  # TODO: should we clone and detach?
        else:
            confidence = None
        return outputs.logits, outputs.loss, confidence
