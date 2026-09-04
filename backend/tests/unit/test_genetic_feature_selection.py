"""
Unit and integration tests for Genetic Algorithm feature selection and its
end-to-end pipeline integration.
"""
from __future__ import annotations

import unittest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression

from backend.services.genetic_selection import GeneticFeatureSelector
from backend.services import preprocessing_service, training_service, dataset_service
from core.state import get_state, new_state_id


class TestGeneticFeatureSelection(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)

    def test_ga_selector_classification_convergence(self):
        """Test GA feature selector on binary classification with informative vs noise features."""
        # 100 samples, 12 features (4 informative, 8 noise)
        X_mat, y_vec = make_classification(
            n_samples=100,
            n_features=12,
            n_informative=4,
            n_redundant=2,
            n_repeated=0,
            random_state=42,
        )
        feature_names = [f"feat_{i}" for i in range(12)]
        X = pd.DataFrame(X_mat, columns=feature_names)
        y = pd.Series(y_vec, name="target")

        progress_calls = []

        def callback(gen, total, score, n_sel):
            progress_calls.append((gen, total, score, n_sel))

        selector = GeneticFeatureSelector(
            problem_type="Classification",
            population_size=12,
            generations=8,
            crossover_rate=0.8,
            mutation_rate=0.1,
            max_features=6,
            parsimony_weight=0.15,
            early_stopping_rounds=4,
            random_state=42,
            progress_callback=callback,
        )

        selector.fit(X, y)

        self.assertIsNotNone(selector.best_mask_)
        self.assertTrue(len(selector.best_features_) > 0)
        self.assertTrue(len(selector.best_features_) <= 8)
        self.assertTrue(len(selector.history_) > 0)
        self.assertTrue(len(progress_calls) > 0)
        self.assertGreater(selector.best_score_, -1.0)
        self.assertEqual(len(selector.best_features_), selector.best_mask_.sum())

    def test_ga_selector_regression(self):
        """Test GA selector on continuous regression target."""
        X_mat, y_vec = make_regression(
            n_samples=80,
            n_features=10,
            n_informative=3,
            random_state=42,
        )
        feature_names = [f"reg_feat_{i}" for i in range(10)]
        X = pd.DataFrame(X_mat, columns=feature_names)
        y = pd.Series(y_vec, name="y_val")

        selector = GeneticFeatureSelector(
            problem_type="Regression",
            population_size=10,
            generations=6,
            max_features=5,
            parsimony_weight=0.1,
            random_state=42,
        )

        selector.fit(X, y)

        self.assertIsNotNone(selector.best_mask_)
        self.assertTrue(len(selector.best_features_) > 0)
        self.assertTrue(all(f in feature_names for f in selector.best_features_))

    def test_ga_selector_single_feature_guard(self):
        """Test GA selector when dataset has only 1 feature."""
        X = pd.DataFrame({"single_col": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
        y = pd.Series([0, 0, 0, 1, 1, 1])

        selector = GeneticFeatureSelector(problem_type="Classification")
        selector.fit(X, y)

        self.assertEqual(selector.best_features_, ["single_col"])
        self.assertEqual(len(selector.best_mask_), 1)

    def test_preprocessing_service_with_ga_selection(self):
        """Test _feature_selection function with method='genetic'."""
        X_mat, y_vec = make_classification(
            n_samples=90,
            n_features=10,
            n_informative=3,
            random_state=42,
        )
        columns = [f"col_{i}" for i in range(10)]
        X = pd.DataFrame(X_mat, columns=columns)
        y = pd.Series(y_vec)

        X_sel, sel_cols, info = preprocessing_service._feature_selection(
            X=X,
            y=y,
            method="genetic",
            max_features=5,
            threshold=0.05,
            problem_type="Classification",
            extra_params={"population_size": 10, "generations": 5, "early_stopping_rounds": 3},
        )

        self.assertEqual(info["method"], "genetic")
        self.assertTrue(len(sel_cols) > 0)
        self.assertTrue(len(sel_cols) <= 10)
        self.assertEqual(X_sel.columns.tolist(), sel_cols)
        self.assertIn("ga_history", info)
        self.assertIn("best_fitness", info)

    def test_end_to_end_pipeline_with_ga_and_training(self):
        """
        Verify end-to-end flow:
        Upload dataset -> Run preprocessing with GA -> State registry -> Model training -> Prediction.
        """
        from unittest.mock import patch

        # 1. Setup mock dataset
        n_samples = 100
        n_features = 12
        X_mat, y_vec = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=4,
            random_state=42,
        )
        df = pd.DataFrame(X_mat, columns=[f"feature_{i}" for i in range(n_features)])
        df["target"] = y_vec

        dataset_id = "test_ga_dataset_123"

        # 2. Run preprocessing with Genetic Algorithm feature selection
        preprocess_config = {
            "dataset_id": dataset_id,
            "target_column": "target",
            "problem_type": "Classification",
            "scaling_method": "standard",
            "imputation_strategy": "mean",
            "apply_encoding": False,
            "test_size": 0.2,
            "random_state": 42,
            "feature_selection": {
                "method": "genetic",
                "max_features": 6,
                "population_size": 12,
                "generations": 6,
                "early_stopping_rounds": 3,
            },
        }

        with patch("backend.services.dataset_service.get_dataset", return_value=df):
            res = preprocessing_service.run_preprocessing(preprocess_config)

        self.assertTrue(res["success"])
        state_id = res["state_id"]
        self.assertIsNotNone(state_id)
        self.assertIsNotNone(res["feature_selection_info"])
        self.assertEqual(res["feature_selection_info"]["method"], "genetic")

        state = get_state(state_id)
        self.assertIn("X_train", state)
        self.assertIn("X_test", state)
        self.assertIn("y_train", state)
        self.assertIn("y_test", state)
        self.assertIn("feature_names", state)

        selected_features = state["feature_names"]
        self.assertTrue(len(selected_features) > 0)
        self.assertEqual(state["X_train"].shape[1], len(selected_features))
        self.assertEqual(state["X_test"].shape[1], len(selected_features))

        # 3. Verify downstream training service functions flawlessly with GA-selected features
        train_res = training_service.train(
            X_train=state["X_train"],
            y_train=state["y_train"],
            X_test=state["X_test"],
            y_test=state["y_test"],
            model_type="RandomForest",
            problem_type="Classification",
            hyperparams={"n_estimators": 20, "max_depth": 5},
            cv_method="stratified",
            cv_folds=3,
        )

        self.assertTrue(train_res["success"])
        self.assertIn("metrics", train_res)
        self.assertIn("accuracy", train_res["metrics"])
        self.assertIsNotNone(train_res["model_id"])


if __name__ == "__main__":
    unittest.main()
