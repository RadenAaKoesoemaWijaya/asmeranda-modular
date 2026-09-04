"""
Genetic Algorithm feature selector module.

Implements an evolutionary metaheuristic wrapper for feature selection
optimized for low latency, robust multi-objective scoring, and seamless
integration into the Asmeranda preprocessing pipeline.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

logger = logging.getLogger("asmeranda.services.genetic_selection")


class GeneticFeatureSelector:
    """
    Genetic Algorithm Feature Selector.

    Searches the 2^N combinatorial feature space using evolutionary principles:
    - Binary chromosome representation (1 = feature active, 0 = inactive)
    - Multi-objective fitness function (predictive performance + parsimony penalty)
    - Elitism, tournament selection, uniform crossover, and adaptive bit-flip mutation
    - Early stopping and lightweight surrogate models for high efficiency.
    """

    def __init__(
        self,
        problem_type: str = "Classification",
        population_size: int = 30,
        generations: int = 20,
        crossover_rate: float = 0.8,
        mutation_rate: Optional[float] = None,
        max_features: Optional[int] = None,
        parsimony_weight: float = 0.1,
        early_stopping_rounds: int = 5,
        random_state: int = 42,
        progress_callback: Optional[Callable[[int, int, float, int], None]] = None,
    ):
        self.problem_type = problem_type
        self.population_size = max(6, int(population_size))
        self.generations = max(1, int(generations))
        self.crossover_rate = float(np.clip(crossover_rate, 0.0, 1.0))
        self.mutation_rate = mutation_rate
        self.max_features = max_features if (max_features is not None and max_features > 0) else None
        self.parsimony_weight = float(np.clip(parsimony_weight, 0.0, 0.5))
        self.early_stopping_rounds = max(1, int(early_stopping_rounds))
        self.random_state = random_state
        self.progress_callback = progress_callback

        self.best_mask_: Optional[np.ndarray] = None
        self.best_features_: List[str] = []
        self.best_score_: float = -np.inf
        self.history_: List[Dict[str, Any]] = []

    def _get_surrogate_estimator(self, n_classes: int) -> Tuple[Any, str]:
        """Returns a fast surrogate model and scoring metric suited for problem type."""
        is_classification = "class" in self.problem_type.lower()
        if is_classification:
            if n_classes > 2:
                estimator = DecisionTreeClassifier(max_depth=4, random_state=self.random_state)
                scoring = "accuracy"
            else:
                estimator = LogisticRegression(
                    solver="liblinear",
                    max_iter=100,
                    random_state=self.random_state,
                )
                scoring = "accuracy"
        else:
            estimator = Ridge(alpha=1.0, random_state=self.random_state)
            scoring = "r2"

        return estimator, scoring

    def _evaluate_individual(
        self,
        ind: np.ndarray,
        X: pd.DataFrame,
        y: pd.Series,
        estimator: Any,
        scoring: str,
        n_features: int,
    ) -> float:
        """Evaluates an individual chromosome fitness score."""
        selected_idx = np.where(ind == 1)[0]
        k = len(selected_idx)
        if k == 0:
            return -1e5

        X_sub = X.iloc[:, selected_idx]

        # Fast 3-fold cross validation
        try:
            cv = 3 if len(X) >= 15 else 2
            scores = cross_val_score(estimator, X_sub, y, cv=cv, scoring=scoring, error_score=0.0)
            base_score = float(np.mean(scores))
            if np.isnan(base_score):
                base_score = -1.0
        except Exception as e:
            logger.debug("Cross validation evaluation error in GA individual: %s", e)
            base_score = -1.0

        # Parsimony component: reward selecting fewer features
        parsimony_bonus = (1.0 - (k / n_features)) * self.parsimony_weight

        # Soft penalty if exceeding max_features constraint
        penalty = 0.0
        if self.max_features is not None and k > self.max_features:
            penalty = 0.15 * ((k - self.max_features) / n_features)

        fitness = ((1.0 - self.parsimony_weight) * base_score) + parsimony_bonus - penalty
        return float(fitness)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> GeneticFeatureSelector:
        """
        Runs the genetic algorithm search for the optimal feature subset.
        """
        n_samples, n_features = X.shape
        if n_features <= 1:
            self.best_mask_ = np.ones(n_features, dtype=int)
            self.best_features_ = X.columns.tolist()
            self.best_score_ = 1.0
            return self

        rng = np.random.RandomState(self.random_state)
        n_classes = y.nunique() if ("class" in self.problem_type.lower() and hasattr(y, "nunique")) else 0
        estimator, scoring = self._get_surrogate_estimator(n_classes)

        # Mutation probability (default 1 / n_features, bounded)
        p_mut = self.mutation_rate
        if p_mut is None:
            p_mut = float(np.clip(1.0 / n_features, 0.02, 0.25))

        # 1. Initialize population with intelligent distribution
        pop = np.zeros((self.population_size, n_features), dtype=int)
        
        # 70% random binary masks
        num_random = int(self.population_size * 0.7)
        for i in range(num_random):
            density = rng.uniform(0.15, 0.45)
            pop[i] = rng.binomial(1, density, size=n_features)

        # 20% focused around top variance or correlation
        num_guided = int(self.population_size * 0.2)
        try:
            variances = X.var().fillna(0).values
            top_var_idx = np.argsort(variances)[::-1]
            k_target = min(self.max_features or max(3, n_features // 3), n_features)
            for i in range(num_random, num_random + num_guided):
                pop[i, top_var_idx[:k_target]] = 1
                noise = rng.binomial(1, 0.1, size=n_features)
                pop[i] = np.bitwise_or(pop[i], noise)
        except Exception:
            for i in range(num_random, num_random + num_guided):
                pop[i] = rng.binomial(1, 0.3, size=n_features)

        # Remaining minimal seeds
        for i in range(num_random + num_guided, self.population_size):
            chosen = rng.choice(n_features, size=min(2, n_features), replace=False)
            pop[i, chosen] = 1

        # Ensure all individuals have at least 1 feature
        for ind in pop:
            if ind.sum() == 0:
                ind[rng.randint(0, n_features)] = 1

        self.history_ = []
        self.best_score_ = -np.inf
        self.best_mask_ = pop[0].copy()
        no_improvement_count = 0

        # 2. Generational Evolution Loop
        for gen in range(self.generations):
            fitness_scores = np.zeros(self.population_size)
            for idx in range(self.population_size):
                fitness_scores[idx] = self._evaluate_individual(
                    pop[idx], X, y, estimator, scoring, n_features
                )

            best_gen_idx = int(np.argmax(fitness_scores))
            best_gen_fitness = float(fitness_scores[best_gen_idx])

            if best_gen_fitness > self.best_score_:
                self.best_score_ = best_gen_fitness
                self.best_mask_ = pop[best_gen_idx].copy()
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            n_sel = int(self.best_mask_.sum())
            self.history_.append({
                "generation": gen + 1,
                "best_fitness": round(float(self.best_score_), 4),
                "generation_best": round(best_gen_fitness, 4),
                "n_features": n_sel,
            })

            if self.progress_callback:
                try:
                    self.progress_callback(gen + 1, self.generations, self.best_score_, n_sel)
                except Exception as e:
                    logger.debug("Progress callback exception: %s", e)

            # Early stopping check
            if no_improvement_count >= self.early_stopping_rounds and (gen + 1) >= 6:
                logger.info(
                    "Genetic Algorithm converged early at generation %d (best fitness: %.4f)",
                    gen + 1,
                    self.best_score_,
                )
                break

            # 3. Selection & Reproduction
            new_population = []
            
            # Elitism: retain top 2 best individuals
            sorted_indices = np.argsort(fitness_scores)[::-1]
            new_population.append(pop[sorted_indices[0]].copy())
            if self.population_size > 1:
                new_population.append(pop[sorted_indices[1]].copy())

            # Fill remaining slots using Tournament Selection + Crossover + Mutation
            while len(new_population) < self.population_size:
                # Tournament Selection (size = 3)
                cands_1 = rng.randint(0, self.population_size, size=3)
                p1 = pop[cands_1[np.argmax(fitness_scores[cands_1])]]

                cands_2 = rng.randint(0, self.population_size, size=3)
                p2 = pop[cands_2[np.argmax(fitness_scores[cands_2])]]

                # Uniform Crossover
                if rng.rand() < self.crossover_rate:
                    swap_mask = rng.binomial(1, 0.5, size=n_features).astype(bool)
                    child = np.where(swap_mask, p1, p2)
                else:
                    child = p1.copy()

                # Adaptive Bit-flip Mutation
                mutate_mask = rng.rand(n_features) < p_mut
                child[mutate_mask] = 1 - child[mutate_mask]

                # Zero-feature guard
                if child.sum() == 0:
                    child[rng.randint(0, n_features)] = 1

                new_population.append(child)

            pop = np.array(new_population)

        # Finalize selected features
        if self.best_mask_ is None or self.best_mask_.sum() == 0:
            self.best_mask_ = np.ones(n_features, dtype=int)

        self.best_features_ = X.columns[self.best_mask_ == 1].tolist()
        return self
