import torch
from torch import tensor
from torch.nn import functional
from spred.loss import softmax, gold_values
from sklearn.neighbors import NearestNeighbors
from spred.train import BasicTrainer
from spred.loader import CalibrationLoader, BalancedLoader

def init_confidence_extractor(cconfig, config, task, model):
    confidence_extractor_lookup = {'inv_abstain': inv_abstain_prob,
                                   'max_non_abstain': max_nonabstain_prob,
                                   'max_prob': max_prob,
                                   'random': random_confidence,
                                   'normals_gold': normals_gold_conf}
    name = cconfig['name']
    if name in confidence_extractor_lookup:
        return confidence_extractor_lookup[name]
    elif name == 'mc_dropout':
        return MCDropoutConfidence()
    elif name == 'posttrained':
        return PosttrainedConfidence(task, config, model)
    elif name == 'trustscore':
        return TrustScore(task.train_loader, model, k=10, alpha=.25)
    else:
        raise Exception('Confidence extractor not recognized: {}'.format(name))


def inv_abstain_prob(batch, model=None):
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


def normals_gold_conf(batch, model=None):
    # TRY AGAIN: this isn't the gold confidence for normals
    inputs = batch['inputs']['inputs']
    inputs = inputs[:,:2]
    adjusted = inputs**2
    adjusted = torch.stack([adjusted[:,0] / 0.5, adjusted[:,1] / 0.1])
    return adjusted.sum(dim=0)


class MCDropoutConfidence:
    def __init__(self, n_forward_passes=30, combo_fn=torch.mean):
        self.n_forward_passes = n_forward_passes
        self.combo_fn = combo_fn

    def __call__(self, batch, lite_model):
        output = batch['outputs']
        preds = torch.max(output, dim=1).indices
        pred_probs = []
        for _ in range(self.n_forward_passes):
            model_out = lite_model(batch['inputs'])
            dropout_output = softmax(model_out['outputs'])
            pred_probs.append(gold_values(dropout_output, preds))
        pred_probs = torch.stack(pred_probs)
        confs = self.combo_fn(pred_probs, dim=0)
        return confs


class PosttrainedConfidence:
    def __init__(self, task, config, base_model):
        calib_trainer = BasicTrainer(config,
                                     BalancedLoader(CalibrationLoader(base_model, task.validation_loader)),
                                     BalancedLoader(CalibrationLoader(base_model, task.train_loader)),
                                     conf_fn=random_confidence)
        self.confidence_model, _ = calib_trainer()

    def __call__(self, batch, model=None):
        self.confidence_model.eval()
        with torch.no_grad():
            calibrator_out = self.confidence_model.lite_forward(batch['inputs'])
            dists = softmax(calibrator_out['outputs'])
            confs = dists[:, -1]
        return confs


def k_nearest_neighbors(t, k):
    array = t.numpy()
    nbrs = NearestNeighbors(n_neighbors=k, algorithm="ball_tree").fit(array)
    _, indices = nbrs.kneighbors(array)
    return torch.tensor(indices)


def distance_to_set(pt, t):
    array = torch.cat([pt.unsqueeze(dim=0), t])
    nbrs = NearestNeighbors(n_neighbors=2, algorithm="ball_tree").fit(array)
    distances, _ = nbrs.kneighbors(array)
    return distances[0, -1]


def high_density_set(t, k, alpha):
    array = t.numpy()
    nbrs = NearestNeighbors(n_neighbors=k, algorithm="ball_tree").fit(array)
    distances, _ = nbrs.kneighbors(array)
    sorted_radii = sorted(enumerate(distances[:, -1]), key=lambda x: x[1])
    point_indices = sorted([pt for (pt, _) in sorted_radii][:int((1.-alpha) * len(sorted_radii))])
    return t[point_indices]


def group_by_label(batch):
    result = dict()
    lbls = batch['labels']
    for value in lbls.unique().numpy():
        mask = lbls == value
        row_indices = tensor(range(len(mask)))[mask]
        points = batch['inputs'][row_indices]
        result[value] = points
    return result


def compute_high_density_sets(batch, k, alpha):
    points_by_label = group_by_label(batch)
    return {lbl: high_density_set(points_by_label[lbl], k, alpha)
            for lbl in points_by_label}


class TrustScore:
    def __init__(self, train_loader, model, k, alpha):
        self.k = k
        self.alpha = alpha
        self.model = model
        full_dataset = None
        for batch in train_loader:
            self.model.eval()
            with torch.no_grad():
                model_out = self.model.embed(batch)
            embedding = {'inputs': model_out['outputs'],
                         'labels': batch['labels']}
            if full_dataset is None:
                full_dataset = embedding
            elif len(full_dataset['inputs']) < 1000:
                for key in full_dataset:
                    full_dataset[key] = torch.cat([full_dataset[key], embedding[key]])
        self.high_density_sets = compute_high_density_sets(full_dataset, k, alpha)

    def __call__(self, batch, lite_model=None):
        output = batch['outputs']
        self.model.eval()
        with torch.no_grad():
            model_out = self.model.embed(batch)
        preds = torch.max(output, dim=1).indices
        confidences = []
        for i, point in enumerate(model_out['outputs']):
            dists = {key: distance_to_set(point, self.high_density_sets[key])
                     for key in self.high_density_sets}
            dist_to_pred_class = max(0.00000001, dists[preds[i].item()])
            other_dists = [dists[key] for key in dists if key != preds[i].item()]
            next_closest_dist = min(other_dists)
            confidences.append(next_closest_dist / dist_to_pred_class)
        return tensor(confidences)