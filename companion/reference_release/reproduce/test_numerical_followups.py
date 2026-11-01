from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from calibration_numerical import solve_calibration
from compare_outputs import compare


class NumericalFollowupTests(unittest.TestCase):
    def test_calibration_rejects_unidentified_slope(self):
        with self.assertRaises(ValueError):
            solve_calibration([0, 1, 0, 1], [.5, .5, .5, .5])

    def test_calibration_score_equations_and_input_preservation(self):
        y = np.array([0, 0, 1, 0, 1, 1, 0, 1])
        p = np.array([.1, .2, .3, .4, .6, .7, .8, .9])
        original = p.copy()
        fit = solve_calibration(y, p)
        np.testing.assert_array_equal(p, original)
        self.assertLess(fit['gradient_max'], 1e-8)
        self.assertLess(fit['alternative_parameter_max_error'], 1e-7)

    def test_output_comparison_rejects_missing_row_prediction_and_choice_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            a, b = Path(directory)/'new.csv', Path(directory)/'reference.csv'
            frame = pd.DataFrame({'id': [1, 2], 'p': [.2, .7], 'selected_C': [.1, 1.]})
            frame.to_csv(b, index=False)
            for field, value in [('p', .2001), ('selected_C', 10.)]:
                changed = frame.copy(); changed.loc[0, field] = value
                changed.to_csv(a, index=False)
                with self.assertRaises(AssertionError):
                    compare(a, b, ['id'], ['selected_C'])
            frame.iloc[:1].to_csv(a, index=False)
            with self.assertRaises(AssertionError):
                compare(a, b, ['id'], ['selected_C'])
            frame.iloc[::-1].to_csv(a, index=False)
            self.assertTrue(compare(a, b, ['id'], ['selected_C'])['passed'])


if __name__ == '__main__':
    unittest.main()
