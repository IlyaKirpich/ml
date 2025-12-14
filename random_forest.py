import numpy as np
from collections import Counter

class DecisionTree:

    class SplitCriteria:
        @staticmethod
        def gini(y):
            p = np.bincount(y) / len(y)
            return 1 - np.sum(p ** 2)

        @staticmethod
        def mse(y):
            return np.var(y)

    class BaseNode:
        def __init__(self, feature_idx=None, threshold=None, left=None, right=None, value=None):
            self.feature_idx = feature_idx
            self.threshold = threshold
            self.left = left
            self.right = right
            self.value = value

        def is_leaf(self):
            return self.value is not None

    class BaseTree:
        def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1,
                     criterion='mse', max_features=None, random_state=None):
            self.max_depth = max_depth
            self.min_samples_split = min_samples_split
            self.min_samples_leaf = min_samples_leaf
            self.criterion = criterion
            self.max_features = max_features
            self.root = None
            self.rng = np.random.RandomState(random_state)

        def _best_split(self, X, y):
            n_samples, n_features = X.shape
            if n_samples < self.min_samples_split:
                return None, None

            features = np.arange(n_features)
            if self.max_features is not None:
                features = self.rng.choice(features, self.max_features, replace=False)

            best_gain = -np.inf
            best_idx, best_thr = None, None

            parent_impurity = (
                DecisionTree.SplitCriteria.mse(y)
                if self.criterion == 'mse'
                else DecisionTree.SplitCriteria.gini(y)
            )

            for f in features:
                values = np.unique(X[:, f])
                if len(values) == 1:
                    continue
                thresholds = (values[:-1] + values[1:]) / 2

                for thr in thresholds:
                    left_mask = X[:, f] <= thr
                    right_mask = ~left_mask

                    if (np.sum(left_mask) < self.min_samples_leaf or
                        np.sum(right_mask) < self.min_samples_leaf):
                        continue

                    y_left, y_right = y[left_mask], y[right_mask]

                    if self.criterion == 'mse':
                        gain = parent_impurity - (
                            len(y_left)/n_samples * DecisionTree.SplitCriteria.mse(y_left) +
                            len(y_right)/n_samples * DecisionTree.SplitCriteria.mse(y_right)
                        )
                    else:
                        gain = parent_impurity - (
                            len(y_left)/n_samples * DecisionTree.SplitCriteria.gini(y_left) +
                            len(y_right)/n_samples * DecisionTree.SplitCriteria.gini(y_right)
                        )

                    if gain > best_gain:
                        best_gain = gain
                        best_idx = f
                        best_thr = thr

            return best_idx, best_thr

        def _build_tree(self, X, y, depth=0):
            if (
                len(y) < self.min_samples_split or
                (self.max_depth is not None and depth >= self.max_depth) or
                len(np.unique(y)) == 1
            ):
                return DecisionTree.BaseNode(value=self._calculate_leaf_value(y))

            best_idx, best_thr = self._best_split(X, y)
            if best_idx is None:
                return DecisionTree.BaseNode(value=self._calculate_leaf_value(y))

            mask = X[:, best_idx] <= best_thr
            left = self._build_tree(X[mask], y[mask], depth + 1)
            right = self._build_tree(X[~mask], y[~mask], depth + 1)

            return DecisionTree.BaseNode(best_idx, best_thr, left, right)

        def fit(self, X, y):
            self.root = self._build_tree(X, y)
            return self

        def _traverse(self, x, node):
            if node.is_leaf():
                return node.value
            if x[node.feature_idx] <= node.threshold:
                return self._traverse(x, node.left)
            return self._traverse(x, node.right)

        def predict(self, X):
            return np.array([self._traverse(x, self.root) for x in X])

        def _calculate_leaf_value(self, y):
            raise NotImplementedError

    class RegressionTree(BaseTree):
        def __init__(self, **kwargs):
            super().__init__(criterion='mse', **kwargs)

        def _calculate_leaf_value(self, y):
            return np.mean(y)


class RandomForestClassifier:
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, max_features='sqrt', random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.trees = []

    def _resolve_max_features(self, n_features):
        if self.max_features == 'sqrt':
            return int(np.sqrt(n_features))
        if self.max_features == 'log2':
            return int(np.log2(n_features))
        if isinstance(self.max_features, int):
            return self.max_features
        return n_features

    def fit(self, X, y):
        X, y = np.asarray(X), np.asarray(y)
        self.classes_ = np.unique(y)
        n_samples, n_features = X.shape
        m = self._resolve_max_features(n_features)
        rng = np.random.RandomState(self.random_state)
        self.trees = []

        for i in range(self.n_estimators):
            idx = rng.choice(n_samples, n_samples, replace=True)
            tree = DecisionTree.ClassificationTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                criterion='gini',
                max_features=m,
                random_state=rng.randint(1e9)
            )
            tree.fit(X[idx], y[idx])
            self.trees.append(tree)
        return self

    def predict(self, X):
        X = np.asarray(X)
        preds = []
        for x in X:
            votes = [tree.predict([x])[0] for tree in self.trees]
            preds.append(Counter(votes).most_common(1)[0][0])
        return np.array(preds)

    def predict_proba(self, X):
        X = np.asarray(X)
        proba = np.zeros((len(X), len(self.classes_)))
        for tree in self.trees:
            preds = tree.predict(X)
            for i, c in enumerate(self.classes_):
                proba[:, i] += (preds == c).astype(float)
        return proba / len(self.trees)

    def score(self, X, y):
        return np.mean(self.predict(X) == y)


class RandomForestRegressor:
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2,
                 min_samples_leaf=5, max_features='sqrt', random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.trees = []

    def _resolve_max_features(self, n_features):
        if self.max_features == 'sqrt':
            return int(np.sqrt(n_features))
        if self.max_features == 'log2':
            return int(np.log2(n_features))
        if isinstance(self.max_features, int):
            return self.max_features
        return n_features

    def fit(self, X, y):
        X, y = np.asarray(X), np.asarray(y)
        n_samples, n_features = X.shape
        m = self._resolve_max_features(n_features)
        rng = np.random.RandomState(self.random_state)
        self.trees = []

        for i in range(self.n_estimators):
            idx = rng.choice(n_samples, n_samples, replace=True)
            tree = DecisionTree.RegressionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=m,
                random_state=rng.randint(1e9)
            )
            tree.fit(X[idx], y[idx])
            self.trees.append(tree)
        return self

    def predict(self, X):
        preds = np.array([tree.predict(X) for tree in self.trees])
        return preds.mean(axis=0)

    def score(self, X, y):
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - ss_res / ss_tot
