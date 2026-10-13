import unittest
import numpy as np
from scipy import sparse
from prepare_classical import fit_preprocessing, transform


class PreprocessingTests(unittest.TestCase):
    def test_training_only_scaling_and_frozen_transform(self):
        train = sparse.csr_matrix([[1., 2., 0.], [4., 1., 0.], [2., 8., 0.]])
        fitted = fit_preprocessing(train, 3)
        np.testing.assert_array_equal(fitted['selected'], [0, 1])
        encoded = transform(train, fitted)
        np.testing.assert_allclose(encoded.mean(axis=0), 0, atol=1e-6)
        np.testing.assert_allclose(encoded.std(axis=0), 1, atol=1e-6)
        frozen = {k: np.copy(v) for k, v in fitted.items()}
        # A gene expressed only in held-out cells must not enter the selected set.
        transform(sparse.csr_matrix([[10., 1., 1000.]]), fitted)
        for k in fitted:
            np.testing.assert_array_equal(fitted[k], frozen[k])

    def test_zero_library_rejected(self):
        with self.assertRaises(ValueError):
            fit_preprocessing(sparse.csr_matrix((2, 3)))


if __name__ == '__main__':
    unittest.main()
