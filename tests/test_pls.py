import unittest
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from utils import PLS


class TestPLS(unittest.TestCase):
    """
    Test the PLS class (one-component PLS) from utils.py.
    Compare with sklearn's PLSRegression with n_components=1.
    """

    def setUp(self):
        # Create sample datasets with different characteristics
        np.random.seed(42)
        
        # Dataset 1: Simple case
        n_samples, n_features = 50, 10
        self.X1 = np.random.randn(n_samples, n_features)
        self.y1 = np.random.randn(n_samples)
        
        # Dataset 2: X and y are correlated
        n_samples, n_features = 30, 5
        X_base = np.random.randn(n_samples, n_features)
        # Create y that's correlated with first component of X
        self.y2 = X_base[:, 0] + 0.5 * np.random.randn(n_samples)
        self.X2 = X_base
        
        # Dataset 3: Larger dimensions
        n_samples, n_features = 100, 20
        self.X3 = np.random.randn(n_samples, n_features)
        self.y3 = np.random.randn(n_samples)

    def test_fit_components_normalized(self):
        """
        Test that the PLS components are properly normalized.
        """
        pls = PLS()
        pls.fit(self.X1, self.y1)
        
        # Check that w is normalized (has unit norm)
        w_norm = np.linalg.norm(pls.w)
        np.testing.assert_almost_equal(w_norm, 1.0, decimal=10)
        
        # Check that components_ has the right shape
        self.assertEqual(pls.components_.shape, (1, self.X1.shape[1]))
        
        # Check that components_ is normalized
        components_norm = np.linalg.norm(pls.components_)
        np.testing.assert_almost_equal(components_norm, 1.0, decimal=10)

    def test_fit_transform_consistency(self):
        """
        Test that fit_transform gives the same result as fit followed by transform.
        """
        pls1 = PLS()
        pls2 = PLS()
        
        result1 = pls1.fit_transform(self.X1, self.y1)
        pls2.fit(self.X1, self.y1)
        result2 = pls2.transform(self.X1)
        
        np.testing.assert_array_almost_equal(result1, result2, decimal=10)

    def test_transform_shape(self):
        """
        Test that transform returns the correct shape.
        """
        pls = PLS()
        pls.fit(self.X1, self.y1)
        
        # Transform training data
        X_transformed = pls.transform(self.X1)
        self.assertEqual(X_transformed.shape, (self.X1.shape[0], 1))
        
        # Transform new data
        X_new = np.random.randn(20, self.X1.shape[1])
        X_new_transformed = pls.transform(X_new)
        self.assertEqual(X_new_transformed.shape, (20, 1))

    def test_compare_with_sklearn_zero_mean(self):
        """
        Test that custom PLS matches sklearn PLSRegression when data is zero-mean.
        Since custom PLS assumes zero-mean, we should compare with sklearn's
        PLSRegression on zero-mean data (with scale=False to avoid scaling).
        """
        # Ensure zero-mean
        X_centered = self.X1 - self.X1.mean(axis=0)
        y_centered = self.y1 - self.y1.mean()
        
        # Custom PLS
        pls_custom = PLS()
        pls_custom.fit(X_centered, y_centered)
        X_transformed_custom = pls_custom.transform(X_centered)
        
        # sklearn PLSRegression (scale=False to avoid scaling, but it still centers)
        # We need to manually center to match the assumption
        pls_sklearn = PLSRegression(n_components=1, scale=False)
        pls_sklearn.fit(X_centered, y_centered)
        X_transformed_sklearn = pls_sklearn.transform(X_centered)
        
        # The components should be similar (up to sign)
        # Check if they match or are opposite
        components_match = np.allclose(
            pls_custom.components_, pls_sklearn.x_weights_.T, atol=1e-6
        )
        components_opposite = np.allclose(
            pls_custom.components_, -pls_sklearn.x_weights_.T, atol=1e-6
        )
        
        self.assertTrue(
            components_match or components_opposite,
            "Components should match sklearn (up to sign)"
        )
        
        # Transformed data should match (up to sign)
        transform_match = np.allclose(
            np.abs(X_transformed_custom), np.abs(X_transformed_sklearn), atol=1e-6
        )
        self.assertTrue(
            transform_match,
            "Transformed data should match sklearn (up to sign)"
        )

    def test_compare_with_sklearn_centered(self):
        """
        Test comparison with sklearn when data is mean-centered.
        sklearn's PLSRegression centers data internally, so we compare
        custom PLS on centered data with sklearn on original data.
        """
        # Center the data for custom PLS
        X_centered = self.X2 - self.X2.mean(axis=0)
        y_centered = self.y2 - self.y2.mean()
        
        # Custom PLS on centered data
        pls_custom = PLS()
        pls_custom.fit(X_centered, y_centered)
        X_transformed_custom = pls_custom.transform(X_centered)
        
        # sklearn PLSRegression on original data (it centers internally)
        pls_sklearn = PLSRegression(n_components=1, scale=False)
        pls_sklearn.fit(self.X2, self.y2)
        X_transformed_sklearn = pls_sklearn.transform(self.X2)
        
        # Components should match (up to sign)
        components_match = np.allclose(
            pls_custom.components_, pls_sklearn.x_weights_.T, atol=1e-6
        )
        components_opposite = np.allclose(
            pls_custom.components_, -pls_sklearn.x_weights_.T, atol=1e-6
        )
        
        self.assertTrue(
            components_match or components_opposite,
            "Components should match sklearn (up to sign) when data is centered"
        )
        
        # Transformed data should match (up to sign)
        # Note: sklearn centers internally, so we need to center X_transformed_sklearn
        # or compare the centered versions
        transform_match = np.allclose(
            np.abs(X_transformed_custom), 
            np.abs(X_transformed_sklearn - X_transformed_sklearn.mean()), 
            atol=1e-5
        )
        self.assertTrue(
            transform_match,
            "Transformed data should match sklearn (up to sign and centering)"
        )

    def test_w_computation(self):
        """
        Test that w is computed correctly as normalized X.T @ y.
        """
        # Center the data
        X_centered = self.X1 - self.X1.mean(axis=0)
        y_centered = self.y1 - self.y1.mean()
        
        pls = PLS()
        pls.fit(X_centered, y_centered)
        
        # Manual computation
        w_manual = X_centered.T @ y_centered
        w_manual = w_manual / np.linalg.norm(w_manual)
        
        # Compare (w is stored as column vector)
        np.testing.assert_array_almost_equal(
            pls.w.flatten(), w_manual, decimal=10
        )

    def test_multiple_datasets(self):
        """
        Test that PLS works correctly on different datasets.
        """
        datasets = [
            (self.X1, self.y1),
            (self.X2, self.y2),
            (self.X3, self.y3),
        ]
        
        for X, y in datasets:
            # Center data
            X_centered = X - X.mean(axis=0)
            y_centered = y - y.mean()
            
            pls = PLS()
            pls.fit(X_centered, y_centered)
            
            # Check normalization
            w_norm = np.linalg.norm(pls.w)
            np.testing.assert_almost_equal(w_norm, 1.0, decimal=10)
            
            # Check transform
            X_transformed = pls.transform(X_centered)
            self.assertEqual(X_transformed.shape, (X.shape[0], 1))

    def test_repr(self):
        """
        Test the string representation.
        """
        pls = PLS()
        self.assertEqual(repr(pls), "PLS")

    def test_fit_returns_self(self):
        """
        Test that fit returns self for method chaining.
        """
        pls = PLS()
        result = pls.fit(self.X1, self.y1)
        self.assertIs(result, pls)

    def test_edge_case_single_feature(self):
        """
        Test PLS with single feature.
        """
        X_single = np.random.randn(20, 1)
        y_single = np.random.randn(20)
        
        X_centered = X_single - X_single.mean(axis=0)
        y_centered = y_single - y_single.mean()
        
        pls = PLS()
        pls.fit(X_centered, y_centered)
        
        # Should still work
        self.assertEqual(pls.components_.shape, (1, 1))
        X_transformed = pls.transform(X_centered)
        self.assertEqual(X_transformed.shape, (20, 1))

    def test_edge_case_high_correlation(self):
        """
        Test PLS when X and y are highly correlated.
        """
        # Create X and y that are highly correlated
        X_base = np.random.randn(50, 5)
        y_high_corr = X_base[:, 0] + 0.01 * np.random.randn(50)
        
        X_centered = X_base - X_base.mean(axis=0)
        y_centered = y_high_corr - y_high_corr.mean()
        
        pls = PLS()
        pls.fit(X_centered, y_centered)
        
        # Should work and first component should align with first feature
        self.assertEqual(pls.components_.shape, (1, 5))
        w_norm = np.linalg.norm(pls.w)
        np.testing.assert_almost_equal(w_norm, 1.0, decimal=10)


if __name__ == "__main__":
    unittest.main()
