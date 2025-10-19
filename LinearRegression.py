import numpy as np

class CrossValidation:

    def kfold_cross_validation(X, y, model, k=5, random_state=None):
        if random_state is not None:
            np.random.seed(random_state)
        
        samples = len(y)
        indices = np.arange(samples)
        
        np.random.shuffle(indices)
        
        fold_size = samples // k
        metrics = {'mse': [], 'r2': [], 'mae': []}
        
        for i in range(k):
            start = i * fold_size
            end = (i + 1) * fold_size if i < k - 1 else samples
            test_indices = indices[start:end]
            
            train_indices = np.concatenate([indices[:start], indices[end:]])
            
            X_train, X_test = X[train_indices], X[test_indices]
            y_train, y_test = y[train_indices], y[test_indices]
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            mse = np.mean((y_test - y_pred)**2)
            mae = np.mean(np.abs(y_test - y_pred))
            ss_res = np.sum((y_test - y_pred)**2)
            ss_tot = np.sum((y_test - np.mean(y_train))**2)
            r2 = 1 - (ss_res / (ss_tot + 1e-10))
            
            metrics['mse'].append(mse)
            metrics['r2'].append(r2)
            metrics['mae'].append(mae)
        
        return {k: (np.mean(v), np.std(v)) for k, v in metrics.items()}

    def loo_cross_validation(X, y, model):
        n_samples = len(y)
        metrics = {'mse': [], 'mae': []}
        
        for i in range(n_samples):
            test_indices = [i]
            train_indices = [j for j in range(n_samples) if j != i]
            
            X_train, X_test = X[train_indices], X[test_indices]
            y_train, y_test = y[train_indices], y[test_indices]
    
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            metrics['mse'].append((y_test[0] - y_pred[0])**2)
            metrics['mae'].append(np.abs(y_test[0] - y_pred[0]))
        
        return {k: (np.mean(v), np.std(v)) for k, v in metrics.items()}


class LinearRegression:

    class LossAndDerivatives:

        @staticmethod
        def mse(X, Y, w, b):
            return np.mean((X.dot(w) + b - Y) ** 2)

        @staticmethod
        def mae(X, Y, w, b):
            return np.mean(np.abs(X.dot(w) + b - Y))

        @staticmethod
        def r2_score(X, Y, w, b):
            return 1 - np.sum((X.dot(w) + b - Y) ** 2) / np.sum((np.mean(Y) - Y) ** 2)

        @staticmethod
        def mape(X, Y, w, b):
            return np.mean(np.abs(X.dot(w) + b - Y) / np.abs(Y))

        @staticmethod
        def l2_reg(w):
            return np.sum(w ** 2)

        @staticmethod
        def l1_reg(w):
            return np.sum(np.abs(w))
        
        @staticmethod
        def lp_reg(w, p):
            return np.sum(np.abs(w) ** p) ** (1 / p)

        @staticmethod
        def no_reg(w):
            return 0.0

        @staticmethod
        def mse_derivative(X, Y, w, b):
            error = X.dot(w) + b - Y
            dw = (2 / X.shape[0]) * X.T.dot(error)
            db = (2 / X.shape[0]) * np.sum(error)
            return dw, db

        @staticmethod
        def mae_derivative(X, Y, w, b):
            error = X.dot(w) + b - Y
            dw = (1 / X.shape[0]) * X.T.dot(np.sign(error))
            db = (1 / X.shape[0]) * np.sum(np.sign(error))
            return dw, db

        @staticmethod
        def l2_reg_derivative(w):
            return 2 * w

        @staticmethod
        def l1_reg_derivative(w):
            return np.sign(w)

        @staticmethod
        def lp_reg_derivative(w, p):
            return p * np.sign(w) * (np.abs(w)) ** (p - 1)

        @staticmethod
        def no_reg_derivative(w):
            return np.zeros_like(w)


    class Normalization:

        @staticmethod
        def z_score(X):
            return (X - np.mean(X)) / np.std(X)

        @staticmethod
        def min_max_scalling(X):
            return (X - np.min(X)) / (np.max(X) - np.min(X))
        

    class LRAnalytical:

        def __init__(self, reg = None, alpha = 0.01):
            self.reg = reg
            self.alpha = alpha
            self.weights = None
            self.bias = 0.0

        def fit(self, X, Y):
            X = np.insert(X, 0, 1, axis=1)
            X_T_X = X.T @ X
            if self.reg == "l2":
                X_T_X += self.alpha * np.eye(X_T_X.shape[0])
            inverse = np.linalg.pinv(X_T_X)
            w = inverse @ X.T @ Y
            self.bias = w[0]
            self.weights = w[1:]

        def predict(self, X):
            return X @ self.weights + self.bias
        

    class LRClassic:

        def __init__(self, loss="mse", reg = None, reg_coef=0.01, learning_rate=0.01, n_iter=1000):
            self.loss = loss
            self.reg = reg
            self.reg_coef = reg_coef
            self.learning_rate = learning_rate
            self.n_iter = n_iter
            self.weights = None
            self.bias = 0.0

        def fit(self, X, Y):
            self.weights = np.zeros(X.shape[1])
            self.bias = 0.0

            if self.loss == 'mse':
                loss = LinearRegression.LossAndDerivatives.mse
                derivative = LinearRegression.LossAndDerivatives.mse_derivative
            elif self.loss == 'mae':
                loss = LinearRegression.LossAndDerivatives.mae
                derivative = LinearRegression.LossAndDerivatives.mae_derivative
            else:
                raise ValueError(f"Loss function {self.loss} doesn't exist.")
            
            if self.reg == 'l1':
                reg = LinearRegression.LossAndDerivatives.l1_reg
                reg_derivative = LinearRegression.LossAndDerivatives.l1_reg_derivative
            elif self.reg == 'l2':
                reg = LinearRegression.LossAndDerivatives.l2_reg
                reg_derivative = LinearRegression.LossAndDerivatives.l2_reg_derivative
            elif self.reg == None:
                reg = LinearRegression.LossAndDerivatives.no_reg
                reg_derivative = LinearRegression.LossAndDerivatives.no_reg_derivative
            else:
                raise ValueError(f"Regularization {self.reg} doesn't exist.")
            
            for i in range(self.n_iter):
                dw, db = derivative(X, Y, self.weights, self.bias)

                dw += self.reg_coef * reg_derivative(self.weights)

                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db

        def predict(self, X):
            return X @ self.weights + self.bias
        
    
    class LRStochastic:

        def __init__(self, loss="mse", reg = None, reg_coef=0.01, learning_rate=0.01, n_iter=1000):
            self.loss = loss
            self.reg = reg
            self.reg_coef = reg_coef
            self.learning_rate = learning_rate
            self.n_iter = n_iter
            self.weights = None
            self.bias = 0.0

        def fit(self, X, Y, batch_size=32, shuffle=True):
            self.weights = np.zeros(X.shape[1])
            self.bias = 0.0

            if self.loss == 'mse':
                loss = LinearRegression.LossAndDerivatives.mse
                derivative = LinearRegression.LossAndDerivatives.mse_derivative
            elif self.loss == 'mae':
                loss = LinearRegression.LossAndDerivatives.mae
                derivative = LinearRegression.LossAndDerivatives.mae_derivative
            else:
                raise ValueError(f"Loss function {self.loss} doesn't exist.")
            
            if self.reg == 'l1':
                reg = LinearRegression.LossAndDerivatives.l1_reg
                reg_derivative = LinearRegression.LossAndDerivatives.l1_reg_derivative
            elif self.reg == 'l2':
                reg = LinearRegression.LossAndDerivatives.l2_reg
                reg_derivative = LinearRegression.LossAndDerivatives.l2_reg_derivative
            elif self.reg == None:
                reg = LinearRegression.LossAndDerivatives.no_reg
                reg_derivative = LinearRegression.LossAndDerivatives.no_reg_derivative
            else:
                raise ValueError(f"Regularization {self.reg} doesn't exist.")
            
            samples = X.shape[0]
            indices = np.arange(samples)

            for epoch in range(self.n_iter):
                if shuffle:
                    np.random.shuffle(indices)
                
                for start_idx in range(0, samples, batch_size):
                    end_idx = min(start_idx + batch_size, samples)
                    batch_indices = indices[start_idx:end_idx]
                    
                    X_batch = X[batch_indices]
                    Y_batch = Y[batch_indices]
                    
                    dw, db = derivative(X_batch, Y_batch, self.weights, self.bias)
                    
                    dw += self.reg_coef * reg_derivative(self.weights)
                    
                    self.weights -= self.learning_rate * dw
                    self.bias -= self.learning_rate * db
            
        def predict(self, X):
            return X @ self.weights + self.bias