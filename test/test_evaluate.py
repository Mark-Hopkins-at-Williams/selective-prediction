import unittest
import numpy as np
import json
from spred.evaluate import Evaluator
from spred.evaluate import KendallTau, PRCurve, GlueMetric
from spred.evaluate import RocCurve, RiskCoverageCurve


def compare(a1, a2, num_decimal_places=4):
    comparison = a1.round(num_decimal_places) == a2.round(num_decimal_places)
    try:
        return comparison.all()
    except Exception:
        return False


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
        assert KendallTau.kendall_tau_distance([1, 2, 3, 4, 5], [0, 0, 1, 1, 1]) == 0
        assert KendallTau.kendall_tau_distance([1, 2, 3, 4, 5], [0, 1, 0, 1, 1]) == 1
        assert KendallTau.kendall_tau_distance([1, 2, 3, 4, 5], [1, 0, 0, 1, 1]) == 2
        assert KendallTau.kendall_tau_distance([1, 2, 3, 4, 5], [1, 0, 1, 0, 1]) == 3
        assert KendallTau.kendall_tau_distance([1, 2, 3, 4, 5], [1, 1, 0, 0, 1]) == 4

    def test_harsh_sort(self):
        assert KendallTau.harsh_sort([1, 2, 3, 4, 5], [0, 0, 1, 1, 1]) == [0, 0, 1, 1, 1]
        assert KendallTau.harsh_sort([3, 4, 5, 1, 2], [1, 1, 1, 0, 0]) == [0, 0, 1, 1, 1]
        assert KendallTau.harsh_sort([1, 2, 2, 3, 5], [0, 0, 1, 1, 1]) == [0, 1, 0, 1, 1]
        assert KendallTau.harsh_sort([2, 3, 5, 1, 2], [1, 1, 1, 0, 0]) == [0, 1, 0, 1, 1]
        assert KendallTau.harsh_sort([2, 2, 2, 2, 2], [1, 1, 1, 0, 0]) == [1, 1, 1, 0, 0]

    def test_kendall_tau_distance_ties(self):
        assert KendallTau.kendall_tau_distance([1, 2, 2, 3, 5], [0, 0, 1, 1, 1]) == 1
        assert KendallTau.kendall_tau_distance([1, 2, 2, 3, 5], [0, 1, 0, 1, 1]) == 1
        assert KendallTau.kendall_tau_distance([2, 2, 2, 2, 2], [0, 1, 0, 1, 1]) == 6

    def test_relativized_kendall_tau_distance(self):
        assert KendallTau.relativized_kendall_tau_distance([1.4, 1.2, 1.3, 1.1], [0, 0, 1, 1]) == 0.75

    def test_glue_metric(self):
        metric = GlueMetric('cola')
        metric.notify_batch(self.preds1)
        assert metric() == {'matthews_correlation': 0.6}

    def test_pr_curve2(self):
        preds = [{'gold': 1, 'pred': 1, 'confidence': 0.1, 'abstain': False},
                 {'gold': 1, 'pred': 2, 'confidence': 0.2, 'abstain': False},
                 {'gold': 1, 'pred': 1, 'confidence': 0.3, 'abstain': False},
                 {'gold': 1, 'pred': 2, 'confidence': 0.4, 'abstain': False},
                 {'gold': 1, 'pred': 2, 'confidence': 0.5, 'abstain': False},
                 {'gold': 1, 'pred': 2, 'confidence': 0.6, 'abstain': False},
                 {'gold': 1, 'pred': 1, 'confidence': 0.7, 'abstain': False},
                 {'gold': 1, 'pred': 1, 'confidence': 0.8, 'abstain': False},
                 {'gold': 1, 'pred': 1, 'confidence': 0.9, 'abstain': False},
                 {'gold': 1, 'pred': 1, 'confidence': 1.0, 'abstain': False}]
        pr_curve = PRCurve()
        pr_curve.notify_batch(preds)
        precision, recall, auc = pr_curve()
        assert compare(precision, np.array([0.6, 0.55555556, 0.625, 0.57142857,
                                            0.66666667, 0.8, 1., 1., 1., 1., 1.]))
        assert compare(recall, np.array([1.0, 0.8333333, 0.8333333, 0.6666667,
                                         0.6666667, 0.6666667, 0.66666667, 0.5,
                                         0.333333, 0.16666667, 0.0]))
        assert approx(auc, 0.86266534)

    def test_pr_curve3(self):
        preds = [{'gold': 1, 'pred': 1, 'confidence': 0.1, 'abstain': False},
                 {'gold': 1, 'pred': 2, 'confidence': 0.2, 'abstain': False},
                 {'gold': 1, 'pred': 1, 'confidence': 0.3, 'abstain': False}]
        pr_curve = PRCurve()
        pr_curve.notify_batch(preds)
        precision, recall, auc = pr_curve()
        assert compare(precision, np.array([0.66666667, 0.5, 1.0, 1.0]))
        assert compare(recall, np.array([1.0, 0.5, 0.5, 0]))
        assert approx(auc, 0.7916666)

    def test_pr_curve4(self):
        preds = [{'gold': 1, 'pred': 1, 'confidence': 0.2, 'abstain': False},
                 {'gold': 1, 'pred': 2, 'confidence': 0.1, 'abstain': False},
                 {'gold': 1, 'pred': 1, 'confidence': 0.3, 'abstain': False}]
        pr_curve = PRCurve()
        pr_curve.notify_batch(preds)
        precision, recall, auc = pr_curve()
        assert compare(precision, np.array([1.0, 1.0, 1.0]))
        assert compare(recall, np.array([1.0, 0.5, 0]))
        assert approx(auc, 1.0)

    def test_pr_curve5(self):
        preds = [{'gold': 1, 'pred': 1, 'confidence': 0.1, 'abstain': False},
                 {'gold': 1, 'pred': 2, 'confidence': 0.3, 'abstain': False},
                 {'gold': 1, 'pred': 1, 'confidence': 0.2, 'abstain': False}]
        pr_curve = PRCurve()
        pr_curve.notify_batch(preds)
        precision, recall, auc = pr_curve()
        assert compare(precision, np.array([0.66666667, 0.5, 0.0, 1.0]))
        assert compare(recall, np.array([1.0, 0.5, 0, 0]))
        assert approx(auc, 0.416666666)

    def test_roc_curve(self):
        curve = RocCurve()
        curve.notify_batch(self.preds1)
        fpr, tpr, auc = curve()
        assert compare(fpr, np.array([0., 0., 0., 0.5, 0.5, 1. ]))
        assert compare(tpr, np.array([0., 0.33333333, 0.66666667, 
                                      0.66666667, 1., 1. ]))
        assert approx(auc, 0.8333333333333333)

    def test_risk_coverage_curve(self):
        curve = RiskCoverageCurve()
        curve.notify_batch(self.preds1)
        coverage, risk, capacity = curve()
        expected_risk = np.array([0.4, 0.25, 0.3333, 0., 0., 0.])
        expected_coverage = np.array([1.0, 0.8, 0.6, 0.4, 0.2, 0.])
        # assert compare(expected_risk, risk)
        # assert compare(expected_coverage, coverage)
        # assert approx(capacity, 0.84333)

    def test_evaluator(self):
        preds = [{'gold': 1, 'pred': 1, 'confidence': 0.1, 'abstain': False},
                 {'gold': 1, 'pred': 2, 'confidence': 0.2, 'abstain': False},
                 {'gold': 1, 'pred': 1, 'confidence': 0.3, 'abstain': False},
                 {'gold': 1, 'pred': 2, 'confidence': 0.4, 'abstain': False},
                 {'gold': 1, 'pred': 2, 'confidence': 0.5, 'abstain': False},
                 {'gold': 1, 'pred': 2, 'confidence': 0.6, 'abstain': False},
                 {'gold': 1, 'pred': 1, 'confidence': 0.7, 'abstain': False},
                 {'gold': 1, 'pred': 1, 'confidence': 0.8, 'abstain': False},
                 {'gold': 1, 'pred': 1, 'confidence': 0.9, 'abstain': False},
                 {'gold': 1, 'pred': 1, 'confidence': 1.0, 'abstain': False}]
        evaluator = Evaluator(preds, task_name="mrpc")
        print(evaluator.get_result())

if __name__ == "__main__":
    unittest.main()
