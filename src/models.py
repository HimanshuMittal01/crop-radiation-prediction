from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import StackingRegressor

class BaseModel:
    def fit(self):
        pass
    def predict(self):
        pass

class StandaloneModel(BaseModel):
    """Standalone models
    Supports only regression problems for now
    """
    AVAILABLE_MODELS = [
        'random_forest',
        'gradient_boosting',
        'ridge',
        'lasso',
        'svr',
        'gaussian_process'
    ]

    def __init__(self, algorithm, problem_type, scoring, random_state=42, **kwargs):
        self.algorithm = algorithm.lower()
        self.problem_type = problem_type.lower()
        self.scoring = scoring
        self.random_state = random_state
        self.params = kwargs

        # Add random state to params
        self.params.update({'random_state': self.random_state})

        # Initialize model
        self.model = self._create_model()

    def _create_model(self):
        model = None
        if self.problem_type=='regression' or self.problem_type=='r':
            if self.algorithm=='random_forest':
                model = RandomForestRegressor(**self.params)
            elif self.algorithm=='gradient_boosting':
                model = GradientBoostingRegressor(**self.params)
            elif self.algorithm=='ridge':
                model = Ridge(**self.params)
            elif self.algorithm=='lasso':
                model = Lasso(**self.params)
            elif self.algorithm=='svr':
                model = SVR(**self.params)
            elif self.algorithm=='gaussian_process':
                model = GaussianProcessRegressor(**self.params)
        else:
            raise Exception(f"{self.problem_type} is not supported yet")

        return model

    def fit(self, train_loader, valid_loader):
        # TODO: Use partial fit if dataset is large
        
        # Load full dataset
        X_train, y_train = train_loader.dataset.get_full_dataset()
        X_valid, y_valid = valid_loader.dataset.get_full_dataset()

        # Train the model
        self._train(X_train, y_train)

        # Evaluate
        valid_eval = self._eval(X_valid, y_valid)
        train_eval = self._eval(X_train, y_train)

        return [train_eval], [valid_eval]

    def _train(self, X, y):
        """
        Does not support multilabel data
        """
        self.model.fit(X, y.ravel())

    def _eval(self, X, y):
        """
        Uses default scorer fn of model if criteria is not provided
        """
        if self.scoring is not None:
            # Predict on X
            y_pred = self.model.predict(X)

            return self.scoring(y, y_pred)
        else:
            return seld.model.score(X, y)