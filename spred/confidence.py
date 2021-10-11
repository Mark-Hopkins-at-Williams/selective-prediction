import torch
from abc import ABC, abstractmethod
from torch import tensor
from torch.nn import functional
from spred.loss import softmax, gold_values
from sklearn.neighbors import NearestNeighbors
from spred.hub import spred_hub


class Confidence:

    def __init__(self):
        self.ident = None

    def train(self, train_loader, model):
        """
        If the confidence needs to be trained, override this method. As
        arguments, you are provided with the training data and the trained
        prediction function.

        """
        pass

    def identifier(self):
        """
        Provides the analytics engine with a short string by which it can refer
        to the confidence function.

        """
        return self.ident

    @abstractmethod
    def __call__(self, batch, model):
        """
        For each prediction in the batch, computes an associated confidence.

        ```batch``` is a dictionary with the following keys:
        - ```inputs```: a torch.tensor of shape BxD, where B is the batch size
          and D is the dimension of the input vectors. Each row corresponds to
          an input instance.
        - ```outputs```: a torch.tensor of shape BxL, where B is the batch size
          and L is the number of labels. Each row corresponds to the predicted
          values for each label. These are not assumed to be normalized.

        This function is expected to return a torch.tensor of shape B,
        containing the confidences of each prediction in the batch.

        """
        ...


class RandomConfidence(Confidence):
    def __call__(self, batch, model=None):
        output = batch['outputs']
        return torch.rand(output.shape[0])


class MaxProb(Confidence):
    def __call__(self, batch, model=None):
        output = batch['outputs']
        print("output")
        print(output)
        probs = functional.softmax(output.clamp(min=-25, max=25), dim=-1)
        return probs.max(dim=1).values


class ProbabilityDifference(Confidence):
    def __call__(self, batch, model=None):
        def second_highest(t):
            indices = t.max(dim=1).indices.unsqueeze(dim=1)
            negate_highest = t.scatter_add(1, indices, -torch.ones(len(t), 1))
            return negate_highest.max(dim=1).values
        output = batch['outputs']
        probs = functional.softmax(output.clamp(min=-25, max=25), dim=-1)
        highest = probs.max(dim=1).values
        next_highest = second_highest(probs)
        return highest-next_highest


class MaxNonabstainProb(Confidence):
    def __call__(self, batch, model=None):
        output = batch['outputs']
        probs = functional.softmax(output.clamp(min=-25, max=25), dim=-1)
        return probs[:, :-1].max(dim=1).values


class SumNonabstainProb(Confidence):
    def __call__(self, batch, model=None):
        output = batch['outputs']
        probs = functional.softmax(output.clamp(min=-25, max=25), dim=-1)
        return 1.0 - probs[:, -1]


class MCDropoutConfidence(Confidence):
    def __init__(self, aggregator, n_forward_passes):
        super().__init__()
        self.n_forward_passes = n_forward_passes
        if aggregator == 'mean':
            self.combo_fn = lambda x: torch.mean(x, dim=0)
            self.ident = "mcdm"
        elif aggregator == 'negvar':
            self.combo_fn = lambda x: -torch.var(x, dim=0)
            self.ident = "mcdv"
        else:
            raise Exception('Aggregator not recognized: {}'.format(aggregator))

    def __call__(self, batch, model):
        output = batch['outputs']
        preds = torch.max(output, dim=1).indices
        pred_probs = []
        for _ in range(self.n_forward_passes):
            model_out = model(batch['inputs'])
            dropout_output = softmax(model_out['outputs'])
            pred_probs.append(gold_values(dropout_output, preds))
        pred_probs = torch.stack(pred_probs)
        confs = self.combo_fn(pred_probs)
        return confs

"""
class PosttrainedConfidence(Confidence):
    def __init__(self, task, config, base_model):
        super().__init__()
        calib_trainer = BasicTrainer(config,
                                     BalancedLoader(CalibrationLoader(base_model, task.init_validation_loader(config['bsz']))),
                                     BalancedLoader(CalibrationLoader(base_model, task.init_train_loader(config['bsz']))),
                                     conf_fn=random_confidence)
        self.confidence_model, _ = calib_trainer()
        self.ident = "pt"

    def __call__(self, batch, model=None):
        self.confidence_model.eval()
        with torch.no_grad():
            calibrator_out = self.confidence_model.lite_forward(batch['inputs'])
            dists = softmax(calibrator_out['outputs'])
            confs = dists[:, -1]
        return confs
"""

class TrustScore(Confidence):
    def __init__(self, k, alpha, max_sample_size=1000):
        super().__init__()
        self.k = k
        self.alpha = alpha
        self.max_sample_size = max_sample_size
        self.ident = "ts" + str(int(alpha*100))
        self.model = None
        self.high_density_sets = None

    def train(self, train_loader, model):
        device = (torch.device("cuda") if torch.cuda.is_available()
                  else torch.device("cpu"))
        full_dataset = None
        self.model = model
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            self.model.eval()
            with torch.no_grad():
                model_out = self.model.embed(batch)
            embedding = {'inputs': model_out['outputs'],
                         'labels': batch['labels']}
            if full_dataset is None:
                full_dataset = embedding
            elif len(full_dataset['inputs']) < self.max_sample_size:
                for key in full_dataset:
                    full_dataset[key] = torch.cat([full_dataset[key], embedding[key]])
            else:
                break
        self.high_density_sets = TrustScore.compute_high_density_sets(full_dataset, self.k, self.alpha)

    @staticmethod
    def distance_to_set(pt, t):
        array = torch.cat([pt.unsqueeze(dim=0), t]).cpu().numpy()
        nbrs = NearestNeighbors(n_neighbors=2, algorithm="ball_tree").fit(array)
        distances, _ = nbrs.kneighbors(array)
        return distances[0, -1]

    @staticmethod
    def high_density_set(t, k, alpha):
        array = t.cpu().numpy()
        nbrs = NearestNeighbors(n_neighbors=k, algorithm="ball_tree").fit(array)
        distances, _ = nbrs.kneighbors(array)
        sorted_radii = sorted(enumerate(distances[:, -1]), key=lambda x: x[1])
        point_indices = sorted([pt for (pt, _) in sorted_radii][:int((1. - alpha) * len(sorted_radii))])
        return t[point_indices]

    @staticmethod
    def group_by_label(batch):
        result = dict()
        lbls = batch['labels']
        for value in lbls.unique().cpu().numpy():
            mask = lbls == value
            row_indices = tensor(range(len(mask)))[mask]
            points = batch['inputs'][row_indices]
            result[value] = points
        return result

    @staticmethod
    def compute_high_density_sets(batch, k, alpha):
        points_by_label = TrustScore.group_by_label(batch)
        return {lbl: TrustScore.high_density_set(points_by_label[lbl], k, alpha)
                for lbl in points_by_label}

    def __call__(self, batch, lite_model=None):
        output = batch['outputs']
        self.model.eval()
        with torch.no_grad():
            model_out = self.model.embed(batch)
        preds = torch.max(output, dim=1).indices
        confidences = []
        for i, point in enumerate(model_out['outputs']):
            dists = {key: TrustScore.distance_to_set(point, self.high_density_sets[key])
                     for key in self.high_density_sets}
            dist_to_pred_class = max(0.00000001, dists[preds[i].item()])
            other_dists = [dists[key] for key in dists if key != preds[i].item()]
            next_closest_dist = min(other_dists)
            confidences.append(next_closest_dist / dist_to_pred_class)
        return tensor(confidences)


spred_hub.register_confidence_fn("random", RandomConfidence)
spred_hub.register_confidence_fn("max_prob", MaxProb)
spred_hub.register_confidence_fn("max_non_abstain", MaxNonabstainProb)
spred_hub.register_confidence_fn("sum_non_abstain", SumNonabstainProb)
spred_hub.register_confidence_fn("mcd", MCDropoutConfidence)
spred_hub.register_confidence_fn("ts", TrustScore)
