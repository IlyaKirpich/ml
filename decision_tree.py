import numpy as np
from collections import Counter


class DecisionTree:
    

    class SplitCriteria:
       

        @staticmethod
        def gini(y):
            proportions = np.bincount(y) / len(y)
            return 1 - np.sum(proportions ** 2)
        

        @staticmethod
        def entropy(y):
            proportions = np.bincount(y) / len(y)
            return -np.sum(p * np.log2(p) for p in proportions if p > 0)
        

        @staticmethod
        def variance(y):
            return np.var(y)
        

        @staticmethod
        def mse(y):
            return np.mean((y - np.mean(y)) ** 2)

    
    class BaseNode:

        
        def __init__(self, feature_idx=None, threshold=None, 
                    left=None, right=None, value=None):
            self.feature_idx = feature_idx    
            self.threshold = threshold   
            self.left = left                 
            self.right = right             
            self.value = value             
        

        def is_leaf(self):
            return self.value is not None
    

    class BaseTree:

        
        def __init__(self, max_depth=None, min_samples_split=2, 
                    min_samples_leaf=1, criterion='gini'):
            self.max_depth = max_depth
            self.min_samples_split = min_samples_split
            self.min_samples_leaf = min_samples_leaf
            self.criterion = criterion
            self.root = None

        
        def _best_split(self, X, y):
            best_gain = -1
            best_idx, best_thresh = None, None
            
            n_samples, n_features = X.shape
            
            if n_samples <= self.min_samples_split:
                return None, None
            
            if self.criterion in ['gini', 'entropy']:
                current_impurity = DecisionTree.SplitCriteria.gini(y) if self.criterion == 'gini' else DecisionTree.SplitCriteria.entropy(y)
            else:
                current_impurity = DecisionTree.SplitCriteria.mse(y)
            
            for feature_idx in range(n_features):
                thresholds = np.unique(X[:, feature_idx])
                
                for threshold in thresholds:
                    left_mask = X[:, feature_idx] <= threshold
                    right_mask = X[:, feature_idx] > threshold
                    
                    if np.sum(left_mask) < self.min_samples_leaf or np.sum(right_mask) < self.min_samples_leaf:
                        continue
                    
                    y_left, y_right = y[left_mask], y[right_mask]
                    
                    if self.criterion in ['gini', 'entropy']:
                        impurity_left = DecisionTree.SplitCriteria.gini(y_left) if self.criterion == 'gini' else DecisionTree.SplitCriteria.entropy(y_left)
                        impurity_right = DecisionTree.SplitCriteria.gini(y_right) if self.criterion == 'gini' else DecisionTree.SplitCriteria.entropy(y_right)
                        n_left, n_right = len(y_left), len(y_right)
                        gain = current_impurity - (n_left / n_samples) * impurity_left - (n_right / n_samples) * impurity_right
                    else:
                        mse_left = DecisionTree.SplitCriteria.mse(y_left)
                        mse_right = DecisionTree.SplitCriteria.mse(y_right)
                        n_left, n_right = len(y_left), len(y_right)
                        gain = current_impurity - (n_left / n_samples) * mse_left - (n_right / n_samples) * mse_right
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_idx = feature_idx
                        best_thresh = threshold
            
            return best_idx, best_thresh

        
        def _build_tree(self, X, y, depth=0):
            n_samples, n_features = X.shape
            
            if (self.max_depth is not None and depth >= self.max_depth or
                n_samples < self.min_samples_split or
                len(np.unique(y)) == 1):
                
                leaf_value = self._calculate_leaf_value(y)
                return DecisionTree.BaseNode(value=leaf_value)
            
            best_idx, best_thresh = self._best_split(X, y)
            
            if best_idx is None:
                leaf_value = self._calculate_leaf_value(y)
                return DecisionTree.BaseNode(value=leaf_value)
            
            left_mask = X[:, best_idx] <= best_thresh
            right_mask = X[:, best_idx] > best_thresh
            
            left_subtree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
            right_subtree = self._build_tree(X[right_mask], y[right_mask], depth + 1)
            
            return DecisionTree.BaseNode(feature_idx=best_idx, threshold=best_thresh,
                                        left=left_subtree, right=right_subtree)
        

        def _calculate_leaf_value(self, y):
            raise NotImplementedError
        
        
        def _traverse_tree(self, x, node):
            if node.is_leaf():
                return node.value
            
            if x[node.feature_idx] <= node.threshold:
                return self._traverse_tree(x, node.left)
            else:
                return self._traverse_tree(x, node.right)
        

        def fit(self, X, y):
            self.root = self._build_tree(X, y)
            return self
        

        def predict(self, X):
            return np.array([self._traverse_tree(x, self.root) for x in X])
    

    class ClassificationTree(BaseTree):

        
        def __init__(self, max_depth=None, min_samples_split=2, 
                    min_samples_leaf=1, criterion='gini'):
            super().__init__(max_depth, min_samples_split, min_samples_leaf, criterion)
        

        def _calculate_leaf_value(self, y):
            counter = Counter(y)
            return counter.most_common(1)[0][0]
        

        def predict_proba(self, X):
            predictions = self.predict(X)
            unique_classes = np.unique(predictions)
            probas = np.zeros((len(X), len(unique_classes)))
            
            for i, pred in enumerate(predictions):
                class_idx = np.where(unique_classes == pred)[0][0]
                probas[i, class_idx] = 1.0
            
            return probas

        
        def score(self, X, y):
            predictions = self.predict(X)
            return np.mean(predictions == y)
    

    class RegressionTree(BaseTree):
        
        def __init__(self, max_depth=None, min_samples_split=2, 
                    min_samples_leaf=1, criterion='mse'):
            super().__init__(max_depth, min_samples_split, min_samples_leaf, criterion)
        

        def _calculate_leaf_value(self, y):
            return np.mean(y)
        

        def score(self, X, y):
            predictions = self.predict(X)
            ss_res = np.sum((y - predictions) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    

    class DecisionTreeClassifier(ClassificationTree):

        
        def __init__(self, max_depth=None, min_samples_split=2, 
                    min_samples_leaf=1, criterion='gini', random_state=None):
            super().__init__(max_depth, min_samples_split, min_samples_leaf, criterion)
            self.random_state = random_state
            if random_state is not None:
                np.random.seed(random_state)
        

        def fit(self, X, y):
            self.classes_ = np.unique(y)
            self.n_classes_ = len(self.classes_)
            return super().fit(X, y)
        

        def predict_proba(self, X):
            predictions = []
            for x in X:
                node = self.root
                while not node.is_leaf():
                    if x[node.feature_idx] <= node.threshold:
                        node = node.left
                    else:
                        node = node.right
                pred = np.zeros(self.n_classes_)
                class_idx = np.where(self.classes_ == node.value)[0][0]
                pred[class_idx] = 1.0
                predictions.append(pred)
            
            return np.array(predictions)

    
    class DecisionTreeRegressor(RegressionTree):
        
        def __init__(self, max_depth=None, min_samples_split=2, 
                    min_samples_leaf=1, criterion='mse', random_state=None):
            super().__init__(max_depth, min_samples_split, min_samples_leaf, criterion)
            self.random_state = random_state
            if random_state is not None:
                np.random.seed(random_state)