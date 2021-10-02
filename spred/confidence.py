import torch
from torch import tensor
from torch.nn import functional
from spred.loss import softmax, gold_values
from sklearn.neighbors import NearestNeighbors
from spred.train import BasicTrainer
from spred.loader import CalibrationLoader, BalancedLoader

def init_confidence_extractor(cconfig, config, task, model):
    confidence_extractor_lookup = {'sum_non_abstain': sum_nonabstain_prob,
                                   'max_non_abstain': max_nonabstain_prob,
                                   'max_prob': max_prob,
                                   'random': random_confidence}
    name = cconfig['name']
    if name in confidence_extractor_lookup:
        return confidence_extractor_lookup[name]
    elif name == 'mcd':
        return MCDropoutConfidence(combo_id=cconfig['aggregator'], n_forward_passes=cconfig['n_forward_passes'])
    elif name == 'posttrained':
        return PosttrainedConfidence(task, config, model)
    elif name == 'ts':
        return TrustScore(task.train_loader, model, k=10, alpha=cconfig['alpha'], max_sample_size=cconfig['max_sample_size'])
    else:
        raise Exception('Confidence extractor not recognized: {}'.format(name))


def sum_nonabstain_prob(batch, model=None):
    output = batch['outputs']
    probs = functional.softmax(output.clamp(min=-25, max=25), dim=-1)
    return 1.0 - probs[:, -1]


def max_nonabstain_prob(batch, model=None):
    output = batch['outputs']
    probs = functional.softmax(output.clamp(min=-25, max=25), dim=-1)
    return probs[:, :-1].max(dim=1).values


def max_prob(batch, model=None):
    output = batch['outputs']
    probs = functional.softmax(output.clamp(min=-25, max=25), dim=-1)
    return probs.max(dim=1).values


def random_confidence(batch, model=None):
    output = batch['outputs']
    return torch.rand(output.shape[0])


class Confidence:

    def __init__(self):
        self.ident = None

    def identifier(self):
        return self.ident


class MCDropoutConfidence(Confidence):
    def __init__(self, combo_id, n_forward_passes):
        super().__init__()
        self.n_forward_passes = n_forward_passes
        self.combo_id = combo_id
        if combo_id == 'mean':
            self.combo_fn = lambda x: torch.mean(x, dim=0)
            self.ident = "mcdm"
        elif combo_id == 'negvar':
            self.combo_fn = lambda x: -torch.var(x, dim=0)
            self.ident = "mcdv"
        else:
            raise Exception('Combo function not recognized: {}'.format(combo_id))

    def __call__(self, batch, model):
        output = batch['outputs']
        preds = torch.max(output, dim=1).indices
        pred_probs = []
        batch = {k: batch[k] for k in batch if batch != output}
        for _ in range(self.n_forward_passes):
            model_out = model(batch)
            dropout_output = softmax(model_out['outputs'])
            pred_probs.append(gold_values(dropout_output, preds))
        pred_probs = torch.stack(pred_probs)
        confs = self.combo_fn(pred_probs)
        return confs


class PosttrainedConfidence(Confidence):
    def __init__(self, task, config, base_model):
        super().__init__()
        calib_trainer = BasicTrainer(config,
                                     BalancedLoader(CalibrationLoader(base_model, task.validation_loader)),
                                     BalancedLoader(CalibrationLoader(base_model, task.train_loader)),
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


class TrustScore(Confidence):
    def __init__(self, train_loader, model, k, alpha, max_sample_size=1000):
        super().__init__()
        self.k = k
        self.alpha = alpha
        self.model = model
        self.ident = "ts" + str(int(alpha*100))
        device = (torch.device("cuda") if torch.cuda.is_available()
                  else torch.device("cpu"))
        full_dataset = None
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            self.model.eval()
            with torch.no_grad():
                model_out = self.model.embed(batch)
            embedding = {'inputs': model_out['outputs'],
                         'labels': batch['labels']}
            if full_dataset is None:
                full_dataset = embedding
            elif len(full_dataset['inputs']) < max_sample_size:
                for key in full_dataset:
                    full_dataset[key] = torch.cat([full_dataset[key], embedding[key]])
            else:
                break
        self.high_density_sets = TrustScore.compute_high_density_sets(full_dataset, k, alpha)

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

