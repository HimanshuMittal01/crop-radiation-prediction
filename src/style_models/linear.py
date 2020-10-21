import numpy as np

class LOWESS:
    """
    Implementation of Locally Weighted Linear Regression

    Parameters
    -------------------------------------
    tau: Controls regularization of local neighbors.
        Lower values lead to overfitting and higher values lead to underfitting
    fit_intercept: True or False, Whether to fit intercept/bias
    online: Learn new examples on-the-fly if True
    """
    def __init__(self, tau=0.67, fit_intercept=True, online=False, random_state=None):
        self.X = None
        self.y = None
        self.tau = tau
        self.fit_intercept = fit_intercept
        self.online = online
        self.random_state = random_state
   
    def fit(self, X, y):
        assert (len(X.shape)==2), "Invalid dimensions"

        self.X = np.array(X)
        self.Y = np.array(y).reshape(-1,1)

        if self.fit_intercept:
            self.X = np.hstack((self.X, np.ones((self.X.shape[0], 1))))

    def predict(self, X):
        # Checks/Asserts
        if self.X is None or self.Y is None:
            raise Exception("Fit the model before calling .predict()")

        X = np.array(X)
        m = len(X)
        if self.fit_intercept:
            X = np.hstack((X, np.ones((m, 1))))
        
        assert (self.X.shape[1]==X.shape[1]), f"Features mismatch {self.X.shape}!={X.shape}"

        y_preds = np.zeros(m)
        for i in range(m):
            M = len(self.X)
            tmpX = np.repeat(np.expand_dims(X[i], axis=0), M, axis=0)
            W = np.exp(np.diag(np.sum((self.X-tmpX)**2, axis=1)) / (-2*self.tau**2))
            theta = np.dot(np.linalg.pinv(np.dot(self.X.T, np.dot(W, self.X))), np.dot(self.X.T, np.dot(W, self.Y)))
            y_preds[i] = np.dot(X[i], theta)
        
        if self.online:
            # Append to training X and y
            # Note: We append after predicting on complete batch for faster runtime
            self.X = np.vstack((self.X, X))
            self.y = np.vstack((self.Y, y_preds.reshape(-1,1)))

        return y_preds