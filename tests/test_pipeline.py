"""Guard the data boundaries and custom transformer contracts."""
import unittest

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from src.data import deduplicate_data, load_combined_data, make_cv, make_split
from src.features import QuantileClipper, WineFeatureEngineer


class PipelineTests(unittest.TestCase):
    def test_split_has_no_duplicate_rows_and_cv_stays_inside_training(self):
        data, removed = deduplicate_data(load_combined_data())
        self.assertEqual(removed, 1177)
        train, test, target, _ = make_split(data)
        train_keys = set(pd.util.hash_pandas_object(train, index=False))
        self.assertFalse(pd.util.hash_pandas_object(test, index=False).isin(train_keys).any())
        validation_indices = []
        for fit_indices, val_indices in make_cv(target):
            self.assertFalse(set(fit_indices) & set(val_indices))
            validation_indices.extend(val_indices)
        self.assertEqual(sorted(validation_indices), list(range(len(train))))

    def test_clipping_learns_only_training_bounds(self):
        training = pd.DataFrame({"value": [0.0, 1.0, 2.0, 3.0]})
        clipper = QuantileClipper(0.25, 0.75).fit(training)
        bounds = clipper.upper_bounds_.copy()
        result = clipper.transform(pd.DataFrame({"value": [-100.0, 100.0]}))
        np.testing.assert_allclose(result.value, [0.75, 2.25])
        pd.testing.assert_series_equal(bounds, clipper.upper_bounds_)
        with self.assertRaises(NotFittedError):
            QuantileClipper().transform(training)
        with self.assertRaises(ValueError):
            QuantileClipper(0.9, 0.1).fit(training)

    def test_feature_engineering_is_cloneable_and_does_not_mutate_input(self):
        data = load_combined_data().drop(columns="quality").head(5)
        original = data.copy(deep=True)
        transformed = clone(WineFeatureEngineer()).fit_transform(data)
        pd.testing.assert_frame_equal(data, original)
        self.assertTrue(np.isfinite(transformed.to_numpy()).all())
        self.assertGreater(len(transformed.columns), len(data.columns))
        pd.testing.assert_frame_equal(WineFeatureEngineer(False).fit_transform(data), data)


if __name__ == "__main__":
    unittest.main()
