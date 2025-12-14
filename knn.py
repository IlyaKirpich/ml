import numpy as np
from collections import Counter
import warnings

class KNeighborsClassifier:
    def __init__(self, n_neighbors=5, weights='uniform', p=2):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.p = p
        
    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        
        if len(self.X_train) < self.n_neighbors:
            warnings.warn(f"n_neighbors ({self.n_neighbors}) > n_samples ({len(self.X_train)})")
            self.n_neighbors = len(self.X_train)
            
        return self
    
    def _compute_distances(self, X):
        distances = []
        for x in X:
            if self.p == 1:
                dist = np.sum(np.abs(self.X_train - x), axis=1)
            elif self.p == 2:
                dist = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
            else:
                dist = np.sum(np.abs(self.X_train - x) ** self.p, axis=1) ** (1/self.p)
            distances.append(dist)
        return np.array(distances)
    
    def predict(self, X):
        X = np.array(X)
        distances = self._compute_distances(X)
        predictions = []
        
        for dist in distances:
            indices = np.argsort(dist)[:self.n_neighbors]
            neighbor_labels = self.y_train[indices]
            
            if self.weights == 'uniform':
                votes = Counter(neighbor_labels)
                prediction = votes.most_common(1)[0][0]
            else:
                weights = 1 / (dist[indices] + 1e-8)
                weighted_votes = {}
                for label, weight in zip(neighbor_labels, weights):
                    weighted_votes[label] = weighted_votes.get(label, 0) + weight
                prediction = max(weighted_votes.items(), key=lambda x: x[1])[0]
            
            predictions.append(prediction)
            
        return np.array(predictions)
    
    def predict_proba(self, X):
        X = np.array(X)
        distances = self._compute_distances(X)
        probabilities = []
        
        for dist in distances:
            indices = np.argsort(dist)[:self.n_neighbors]
            neighbor_labels = self.y_train[indices]
            
            if self.weights == 'uniform':
                weights = np.ones(len(neighbor_labels))
            else:
                weights = 1 / (dist[indices] + 1e-8)
            
            unique_labels = np.unique(self.y_train)
            proba = np.zeros(len(unique_labels))
            
            for i, label in enumerate(unique_labels):
                mask = neighbor_labels == label
                if np.any(mask):
                    proba[i] = np.sum(weights[mask])
            
            proba_sum = np.sum(proba)
            if proba_sum > 0:
                proba = proba / proba_sum
            
            probabilities.append(proba)
            
        return np.array(probabilities)
    
    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)

class KNeighborsRegressor:
    def __init__(self, n_neighbors=5, weights='uniform', p=2):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.p = p
        
    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        
        if len(self.X_train) < self.n_neighbors:
            warnings.warn(f"n_neighbors ({self.n_neighbors}) > n_samples ({len(self.X_train)})")
            self.n_neighbors = len(self.X_train)
            
        return self
    
    def _compute_distances(self, X):
        distances = []
        for x in X:
            if self.p == 1:
                dist = np.sum(np.abs(self.X_train - x), axis=1)
            elif self.p == 2:
                dist = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
            else:
                dist = np.sum(np.abs(self.X_train - x) ** self.p, axis=1) ** (1/self.p)
            distances.append(dist)
        return np.array(distances)
    
    def predict(self, X):
        X = np.array(X)
        distances = self._compute_distances(X)
        predictions = []
        
        for dist in distances:
            indices = np.argsort(dist)[:self.n_neighbors]
            neighbor_values = self.y_train[indices]
            
            if self.weights == 'uniform':
                prediction = np.mean(neighbor_values)
            else:
                weights = 1 / (dist[indices] + 1e-8)
                prediction = np.average(neighbor_values, weights=weights)
            
            predictions.append(prediction)
            
        return np.array(predictions)
    
    def score(self, X, y):
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / (ss_tot + 1e-8))