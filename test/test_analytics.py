import unittest
import numpy as np
import json
from spred.evaluate import Evaluator
from spred.analytics import EvaluationResult, EpochResult
from spred.analytics import ExperimentResult, ResultDatabase


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

    def test_evaluation_result_serialization(self):
        evaluator = Evaluator(self.preds1)
        result = evaluator.get_result().as_dict()
        result = {k: round(result[k], 4) if result[k] is not None else None
                  for k in result}
        expected = {'validation_loss': None,
                    'avg_err_conf': 0.12, 'avg_crr_conf': 0.38,
                    'auroc': 0.8333,
                    'capacity': 0.8433, 'kendall_tau': 0.1667,
                    'accuracy': 0.6}
        assert result == expected
        result2 = EvaluationResult(expected)
        assert result2.as_dict() == expected

    def test_evaluation_result_averaging(self):
        result1 = EvaluationResult({'train_loss': 1, 'avg_err_conf': 2,
                                    'avg_crr_conf': 3, 'auroc': 4})
        result2 = EvaluationResult({'train_loss': 1, 'avg_err_conf': 2,
                                    'avg_crr_conf': 3, 'auroc': 8})
        avg = EvaluationResult.averaged([result1, result2])
        expected = EvaluationResult({'train_loss': 1.0, 'avg_err_conf': 2.0,
                                    'avg_crr_conf': 3.0, 'auroc': 6.0})
        assert avg == expected

    def test_evaluation_result_merge(self):
        result1 = EvaluationResult({'train_loss': 1.0, 'avg_err_conf': 2.0,
                                    'avg_crr_conf': 3.0, 'auroc': 4.6})
        result2 = EvaluationResult({'train_loss': 1.1, 'avg_err_conf': 2.1,
                                    'avg_crr_conf': 3.1, 'auroc': 4.1})
        result3 = EvaluationResult({'train_loss': 1.7, 'avg_err_conf': 2.7,
                                    'avg_crr_conf': 3.7, 'auroc': 4.5})
        result4 = EvaluationResult({'train_loss': 1.8, 'avg_err_conf': 2.8,
                                    'avg_crr_conf': 3.3, 'auroc': 4.8})
        result5 = EvaluationResult({'train_loss': 1.2, 'avg_err_conf': 2.9,
                                    'avg_crr_conf': 3.9, 'auroc': 4.9})
        merged = EvaluationResult.merge([result1, result2, result3, result4, result5])
        expected = {'avg_crr_conf': [3.0, 3.1, 3.7, 3.3, 3.9],
                    'train_loss': [1.0, 1.1, 1.7, 1.8, 1.2],
                    'avg_err_conf': [2.0, 2.1, 2.7, 2.8, 2.9],
                    'auroc': [4.6, 4.1, 4.5, 4.8, 4.9]}
        assert merged == expected
        median = EvaluationResult.median([result1, result2, result3, result4, result5])
        expected = EvaluationResult({'train_loss': 1.2, 'avg_err_conf': 2.7,
                                     'avg_crr_conf': 3.3, 'auroc': 4.6})
        assert median == expected

    def test_epoch_result_serialization(self):
        validation_d = {'train_loss': 1.2, 'avg_err_conf': 0.3, 'avg_crr_conf': 0.6333,
                        'auroc': 0.8333, 'aupr': 0.9028, 'kendall_tau': 0.5,
                        'capacity': 0.94, 'precision': 0.6, 'coverage': 1.0}
        validation = EvaluationResult(validation_d)
        original = EpochResult(3, 0.77, validation)
        d = {'epoch': 3, 'train_loss': 0.77, 'validation_result': validation}
        result = EpochResult.from_dict(d)
        assert result == original

    def test_epoch_result_averaging(self):
        eval_result1 = EvaluationResult({'train_loss': 1, 'avg_err_conf': 2,
                                         'avg_crr_conf': 3, 'auroc': 4})
        eval_result2 = EvaluationResult({'train_loss': 5, 'avg_err_conf': 6,
                                         'avg_crr_conf': 7, 'auroc': 8})
        epoch_results = [EpochResult(3, 1, eval_result1),
                         EpochResult(3, 3, eval_result2)]
        avg = EpochResult.averaged(epoch_results)
        avg_eval_result = EvaluationResult({'train_loss': 3.0, 'avg_err_conf': 4.0,
                                            'avg_crr_conf': 5.0, 'auroc': 6.0})
        expected = EpochResult(3.0, 2.0, avg_eval_result)
        assert avg == expected

    def test_experiment_result_averaging(self):

        def example_result(k):
            epoch_results = []
            validation_d1 = {'dev_loss': k + 1.2, 'kendall_tau': k + 0.4}
            validation = EvaluationResult(validation_d1)
            epoch_results.append(EpochResult(1, k + 0.6, validation))
            validation_d2 = {'dev_loss': k + 0.6, 'kendall_tau': k + 0.2}
            validation = EvaluationResult(validation_d2)
            epoch_results.append(EpochResult(2, k + 0.7, validation))
            return ExperimentResult({'key': 'just an example'}, epoch_results)

        def expected_result():
            epoch_results = []
            validation = EvaluationResult({'dev_loss': 1.8, 'kendall_tau': 1.0})
            epoch_results.append(EpochResult(1, 1.2, validation))
            validation = EvaluationResult({'dev_loss': 1.2, 'kendall_tau': 0.8})
            epoch_results.append(EpochResult(2, 1.3, validation))
            return ExperimentResult({'key': 'just an example'}, epoch_results)

        def close_enough_eval_results(res1, res2):
            assert res1.keys() == res2.keys()
            for key in res1.keys():
                approx(res1[key], res2[key])

        def close_enough_epoch_results(res1, res2):
            res1 = res1.as_dict()
            res2 = res2.as_dict()
            assert res1['epoch'] == res2['epoch']
            assert approx(res1['train_loss'], res2['train_loss'])
            close_enough_eval_results(res1['validation_result'],
                                      res2['validation_result'])

        results = [example_result(0.4), example_result(0.6), example_result(0.8)]
        summarized = ResultDatabase(results).summary()
        expected = expected_result()
        assert summarized.config == expected.config
        for x, y in zip(summarized.epoch_results, expected.epoch_results):
            close_enough_epoch_results(x,y)


if __name__ == "__main__":
    unittest.main()
