"""
Combinatorial Purged Cross-Validation (Lopez de Prado 2018).

Standard k-fold CV leaks information when samples are serially correlated
(financial time series always are).  CPCV addresses this by:

  1. Purging:  removing training observations within `purge_days` of any test
               observation, to prevent label overlap.
  2. Embargo:  excluding `embargo_days` of training data *after* the test set
               to prevent lookahead from post-test observations in features.
  3. Combinatorial:  generates C(n_splits, test_splits) unique test sets,
               giving a full distribution of back-tested paths rather than
               one single path (single-path overfitting).

Usage
-----
>>> cv = CombPurgedCV(n_splits=6, test_splits=2, purge_days=5, embargo_days=5)
>>> for train_idx, test_idx in cv.split(X):
...     model.fit(X[train_idx], y[train_idx])
...     score = model.score(X[test_idx], y[test_idx])
"""

import itertools
import numpy as np
from typing import Generator, List, Tuple


class CombPurgedCV:
    """
    Combinatorial Purged Cross-Validation for time-series data.

    Parameters
    ----------
    n_splits   : Total number of time-series folds (k in k-fold).
    test_splits: Number of folds reserved for testing in each combination.
                 Generates C(n_splits, test_splits) train/test pairs.
    purge_days : Days of training data to purge before each test block.
    embargo_days: Days of training data to exclude after each test block.
    """

    def __init__(
        self,
        n_splits: int = 6,
        test_splits: int = 1,
        purge_days: int = 5,
        embargo_days: int = 5,
    ):
        if test_splits >= n_splits:
            raise ValueError("test_splits must be < n_splits")
        self.n_splits = n_splits
        self.test_splits = test_splits
        self.purge_days = purge_days
        self.embargo_days = embargo_days

    def split(
        self, X, y=None, groups=None
    ) -> Generator[Tuple[List[int], List[int]], None, None]:
        """
        Generate (train_indices, test_indices) pairs.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) — only n_samples is used.

        Yields
        ------
        (train_idx, test_idx) : lists of integer indices.
        """
        n = len(X)
        fold_size = n // self.n_splits
        # Build contiguous folds (last fold absorbs remainder)
        folds = []
        for i in range(self.n_splits):
            start = i * fold_size
            end = (i + 1) * fold_size if i < self.n_splits - 1 else n
            folds.append(list(range(start, end)))

        # Enumerate all C(n_splits, test_splits) test-fold combinations
        for test_fold_ids in itertools.combinations(range(self.n_splits), self.test_splits):
            test_set = set()
            for f in test_fold_ids:
                test_set.update(folds[f])
            test_idx = sorted(test_set)

            if not test_idx:
                continue

            test_start = test_idx[0]
            test_end = test_idx[-1]

            # Collect training indices from non-test folds, with purge + embargo
            train_set = set()
            for f in range(self.n_splits):
                if f in test_fold_ids:
                    continue
                for i in folds[f]:
                    # Purge: too close BEFORE test window
                    if test_start - self.purge_days <= i < test_start:
                        continue
                    # Embargo: too close AFTER test window
                    if test_end < i <= test_end + self.embargo_days:
                        continue
                    train_set.add(i)

            train_idx = sorted(train_set)
            if len(train_idx) >= 10 and len(test_idx) >= 1:
                yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """Number of train/test splits."""
        from math import comb
        return comb(self.n_splits, self.test_splits)
