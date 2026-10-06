import unittest
import pandas as pd
from replay_status import classify_replay


class ReproductionStatusTests(unittest.TestCase):
    def frame(self, c=100, rank=4):
        return pd.DataFrame([dict(C=c, rank_new=rank, rank_old=rank, valid_new=True,
                                 valid_old=True, pooled_logloss_new=.80003, pooled_logloss_old=.8)])

    def evaluate(self, frame=None, strict=False, predictions=True):
        return classify_replay([{'passed': predictions}], self.frame() if frame is None else frame,
                               {'ll': 1e-12}, [], strict)[0]

    def test_documented_intermediate_difference_remains_visible(self):
        r = self.evaluate()
        self.assertTrue(r['command_completed'])
        self.assertFalse(r['passed'])
        self.assertFalse(r['intermediate_scores_passed'])
        self.assertEqual(r['intermediate_mismatch_count'], 1)

    def test_strict_mode_rejects_the_same_difference(self):
        self.assertFalse(self.evaluate(strict=True)['command_completed'])

    def test_changed_predictions_are_never_a_warning(self):
        self.assertFalse(self.evaluate(predictions=False)['command_completed'])

    def test_selected_or_other_candidate_difference_fails(self):
        self.assertFalse(self.evaluate(self.frame(rank=1))['command_completed'])
        self.assertFalse(self.evaluate(self.frame(c=10))['command_completed'])

    def test_changed_rank_or_invalid_score_fails(self):
        f = self.frame(); f.loc[0, 'rank_new'] = 3
        self.assertFalse(self.evaluate(f)['command_completed'])
        f = self.frame(); f.loc[0, 'pooled_logloss_new'] = float('nan')
        self.assertFalse(self.evaluate(f)['command_completed'])


if __name__ == '__main__':
    unittest.main()
