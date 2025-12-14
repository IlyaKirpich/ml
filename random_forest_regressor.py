import numpy as np
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

class RandomForestRegressor:
    """
    Собственная имплементация Random Forest для регрессии.
    
    Parameters:
    -----------
    n_estimators : int, default=100
        Количество деревьев в лесу
    max_depth : int, default=None
        Максимальная глубина деревьев
    min_samples_split : int, default=2
        Минимальное количество образцов для разбиения узла
    min_samples_leaf : int, default=1
        Минимальное количество образцов в листе
    max_features : str, int, float, default='sqrt'
        Количество признаков для рассмотрения при разбиении
    bootstrap : bool, default=True
        Использовать ли bootstrap-выборки
    oob_score : bool, default=False
        Вычислять ли OOB score
    random_state : int, default=None
        Seed для воспроизводимости
    """

    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, max_features='sqrt', bootstrap=True,
                 oob_score=False, random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.random_state = random_state
        
        self.trees = []
        self.n_features_ = None
        self.feature_importances_ = None
        self.oob_score_ = None
    
    def _get_max_features(self, n_features):
        """Определение количества признаков для рассмотрения"""
        if self.max_features is None:
            return n_features
        elif isinstance(self.max_features, int):
            return min(self.max_features, n_features)
        elif isinstance(self.max_features, float):
            return max(1, int(self.max_features * n_features))
        elif self.max_features == 'sqrt':
            return max(1, int(np.sqrt(n_features)))
        elif self.max_features == 'log2':
            return max(1, int(np.log2(n_features)))
        return n_features
          
    def _bootstrap_sample(self, X, y, rng):
        """Создание bootstrap-выборки"""
        n_samples = X.shape[0]
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        oob_indices = np.array(list(set(range(n_samples)) - set(indices)))
        return X[indices], y[indices], oob_indices
      
    def fit(self, X, y):
        """Обучение Random Forest"""
        X = np.array(X)
        y = np.array(y).astype(float)
        
        n_samples, n_features = X.shape
        self.n_features_ = n_features
        
        rng = np.random.RandomState(self.random_state)
        
        self.trees = []
        oob_predictions = np.zeros(n_samples)
        oob_counts = np.zeros(n_samples)
        
        max_f = self._get_max_features(n_features)
        
        for i in range(self.n_estimators):
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=max_f,
                random_state=rng.randint(0, 10000)
            )
            
            if self.bootstrap:
                X_boot, y_boot, oob_idx = self._bootstrap_sample(X, y, rng)
            else:
                X_boot, y_boot, oob_idx = X, y, np.array([])
           

            tree.fit(X_boot, y_boot)
            self.trees.append(tree)
            
            if self.oob_score and len(oob_idx) > 0:
                oob_pred = tree.predict(X[oob_idx])
                oob_predictions[oob_idx] += oob_pred
                oob_counts[oob_idx] += 1
        
        if self.oob_score:
            valid_idx = oob_counts > 0
            if np.sum(valid_idx) > 0:
                oob_predictions[valid_idx] /= oob_counts[valid_idx]
                self.oob_score_ = r2_score(y[valid_idx], oob_predictions[valid_idx])
        
        self._compute_feature_importances()
        
        return self
    
    def _compute_feature_importances(self):
        """Вычисление важности признаков (усреднение по всем деревьям)"""
        importances = np.zeros(self.n_features_)
        for tree in self.trees:
            importances += tree.feature_importances_
        self.feature_importances_ = importances / len(self.trees)
    
    def predict(self, X):
        """Предсказание значений (усреднение по всем деревьям)"""
        X = np.array(X)
        all_predictions = np.zeros((X.shape[0], len(self.trees)))
        
        for i, tree in enumerate(self.trees):
            all_predictions[:, i] = tree.predict(X)
        
        return np.mean(all_predictions, axis=1)