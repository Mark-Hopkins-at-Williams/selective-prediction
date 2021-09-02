import unittest
import numpy as np
from spred.analytics import Evaluator, EvaluationResult, EpochResult
from spred.analytics import ExperimentResult
from spred.analytics import kendall_tau_distance, harsh_sort


def compare(a1, a2, num_decimal_places=4):
    comparison = a1.round(num_decimal_places) == a2.round(num_decimal_places)
    return comparison.all()


def approx(x, y, num_decimal_places=4):
    return round(x, num_decimal_places) == round(y, num_decimal_places)


class TestEvaluator(unittest.TestCase):
    
    def setUp(self):
        self.preds1 = [{'gold': 9, 'pred': 4, 'confidence': 0.1, 'abstain': False},
                       {'gold': 5, 'pred': 5, 'confidence': 0.3, 'abstain': False},
                       {'gold': 7, 'pred': 1, 'confidence': 0.5, 'abstain': False},
                       {'gold': 2, 'pred': 2, 'confidence': 0.7, 'abstain': False},
                       {'gold': 3, 'pred': 3, 'confidence': 0.9, 'abstain': False}]

    def test_kendall_tau_distance(self):
        assert kendall_tau_distance([1, 2, 3, 4, 5], [0, 0, 1, 1, 1]) == 0
        assert kendall_tau_distance([1, 2, 3, 4, 5], [0, 1, 0, 1, 1]) == 1
        assert kendall_tau_distance([1, 2, 3, 4, 5], [1, 0, 0, 1, 1]) == 2
        assert kendall_tau_distance([1, 2, 3, 4, 5], [1, 0, 1, 0, 1]) == 3
        assert kendall_tau_distance([1, 2, 3, 4, 5], [1, 1, 0, 0, 1]) == 4

    def test_harsh_sort(self):
        assert harsh_sort([1, 2, 3, 4, 5], [0, 0, 1, 1, 1]) == [0, 0, 1, 1, 1]
        assert harsh_sort([3, 4, 5, 1, 2], [1, 1, 1, 0, 0]) == [0, 0, 1, 1, 1]
        assert harsh_sort([1, 2, 2, 3, 5], [0, 0, 1, 1, 1]) == [0, 1, 0, 1, 1]
        assert harsh_sort([2, 3, 5, 1, 2], [1, 1, 1, 0, 0]) == [0, 1, 0, 1, 1]
        assert harsh_sort([2, 2, 2, 2, 2], [1, 1, 1, 0, 0]) == [1, 1, 1, 0, 0]

    def test_kendall_tau_distance_ties(self):
        assert kendall_tau_distance([1, 2, 2, 3, 5], [0, 0, 1, 1, 1]) == 1
        assert kendall_tau_distance([1, 2, 2, 3, 5], [0, 1, 0, 1, 1]) == 1
        assert kendall_tau_distance([2, 2, 2, 2, 2], [0, 1, 0, 1, 1]) == 6

    def test_relativized_kendall_tau_distance(self):
        assert kendall_tau_distance([1.4, 1.2, 1.3, 1.1], [0, 0, 1, 1]) == 0.75

    def test_pr_curve(self):
        evaluator = Evaluator(self.preds1)
        precision, recall, auc = evaluator.pr_curve()
        assert compare(precision, np.array([0.6, 0.75, 0.66666667, 1., 1., 1.]))
        assert compare(recall, np.array([1., 1., 0.66666667, 0.66666667,
                                         0.33333333, 0.]))
        assert approx(auc, 0.9027777777777777)
        
    def test_roc_curve(self):
        evaluator = Evaluator(self.preds1)
        fpr, tpr, auc = evaluator.roc_curve()
        assert compare(fpr, np.array([0., 0., 0., 0.5, 0.5, 1. ]))
        assert compare(tpr, np.array([0., 0.33333333, 0.66666667, 
                                      0.66666667, 1., 1. ]))
        assert approx(auc, 0.8333333333333333)

    def test_risk_coverage_curve(self):
        evaluator = Evaluator(self.preds1)
        coverage, risk, capacity = evaluator.risk_coverage_curve()
        expected_risk = np.array([0.2, 0.2, 0., 0., 0.])
        expected_coverage = np.array([0.8, 0.6, 0.4, 0.2, 0.])
        assert compare(expected_risk, risk)
        assert compare(expected_coverage, coverage)
        assert approx(capacity, 0.94)

    """
    def test_evaluation_result_serialization(self):
        evaluator = Evaluator(self.preds1)
        result = evaluator.get_result().as_dict()
        result = {k: round(result[k], 4) if result[k] is not None else None
                  for k in result}
        expected = {'train_loss': None, 'avg_err_conf': 0.3, 'avg_crr_conf': 0.6333,
                    'auroc': 0.8333, 'aupr': 0.9028,
                    'capacity': 0.94, 'precision': 0.6, 'coverage': 1.0}
        assert result == expected
        result2 = EvaluationResult.from_dict(expected)
        assert result2.as_dict() == expected

    def test_epoch_result_serialization(self):
        validation_d = {'train_loss': 1.2, 'avg_err_conf': 0.3, 'avg_crr_conf': 0.6333,
                        'auroc': 0.8333, 'aupr': 0.9028,
                        'capacity': 0.94, 'precision': 0.6, 'coverage': 1.0}
        validation = EvaluationResult.from_dict(validation_d)
        result = EpochResult(3, 0.77, validation)
        expected = {'epoch': 3,
                    'train_loss': 0.77,
                    'validation_result': validation_d}
        assert result.as_dict() == expected
        result2 = EpochResult.from_dict(expected)
        assert result2.as_dict() == expected

    def test_experiment_result_serialization(self):
        results = []
        validation_d1 = {'train_loss': 1.2, 'avg_err_conf': 0.3, 'avg_crr_conf': 0.6333,
                         'auroc': 0.8333, 'aupr': 0.9028,
                         'capacity': 0.94, 'precision': 0.6, 'coverage': 1.0}
        validation = EvaluationResult.from_dict(validation_d1)
        results.append(EpochResult(1, 0.65, validation))
        validation_d2 = {'train_loss': 1.2, 'avg_err_conf': 0.3, 'avg_crr_conf': 0.6333,
                         'auroc': 0.86, 'aupr': 0.92,
                         'capacity': 0.94, 'precision': 0.6, 'coverage': 1.0}
        validation = EvaluationResult.from_dict(validation_d2)
        results.append(EpochResult(2, 0.77, validation))
        experiment_result = ExperimentResult({'key': 'just an example'}, results)
        expected = {'config': {'key': 'just an example'},
                    'results': [{'epoch': 1,
                                 'train_loss': 0.65,
                                 'validation_result': validation_d1},
                                {'epoch': 2,
                                 'train_loss': 0.77,
                                 'validation_result': validation_d2}]}
        assert experiment_result.as_dict() == expected
        result2 = ExperimentResult.from_dict(expected)
        assert result2.as_dict() == expected
    """

if __name__ == "__main__":
    unittest.main()
