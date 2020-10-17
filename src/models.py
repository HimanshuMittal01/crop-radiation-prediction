from tqdm import tqdm

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import StackingRegressor

from src.torch_models.dnn import _Core_DNNRegression
from src.optimizer import TorchOptimizer

class BaseModel:
    """Parent class for models"""
    def __init__(self):
        pass

    def fit(self, train_loader, valid_loader):
        # Train the model
        self._train(train_loader)

        # Evaluate
        valid_eval = self._eval(valid_loader)
        train_eval = self._eval(train_loader)

        return [train_eval], [valid_eval]
    
    def _train(self, data_loader):
        """Must implement function"""
        pass

    def _eval(self, data_loader):
        """Must implement function"""
        pass

    def predict(self):
        pass

class StandaloneModel(BaseModel):
    """Standalone models
    Supports only regression problems for now

    Uses parent's .fit() method
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
        super().__init__()
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
    
    def fit(self, train_loader, valid_loader, epochs):
        return super().fit(train_loader, valid_loader)
    
    def _train(self, data_loader):
        """
        Does not support multilabel data
        """
        # TODO: Use partial fit if dataset is large
        # Load full dataset
        X, y = data_loader.dataset.get_full_dataset()

        self.model.fit(X, y.ravel())

    def _eval(self, data_loader):
        """
        Uses default scorer fn of model if criteria is not provided
        """
        # Load full dataset
        X, y = data_loader.dataset.get_full_dataset()

        if self.scoring is not None:
            # Predict on X
            y_pred = self.model.predict(X)

            return self.scoring(y, y_pred)
        else:
            return self.model.score(X, y)

class DNNRegression(BaseModel):
    def __init__(self, problem_type, optimizer, scoring, num_features, num_targets, hidden_layers, device="cuda:0", **kwargs):
        super().__init__()
        self.problem_type = problem_type
        self.scoring = scoring

        self.num_features = num_features
        self.num_targets = num_targets
        self.hidden_layers = hidden_layers

        self.device = device
        self.params = kwargs # Optimizer params

        self.model = self._create_model()
        self.optimizer = self._build_optimizer(optimizer)
    
    def _create_model(self):
        model_params = {}
        for k,v in self.params.items():
            if k in ['dropout_rate']:
                model_params[k] = v
        
        model = _Core_DNNRegression(
            D_in=self.num_features,
            H=self.hidden_layers,
            D_out=self.num_targets,
            **model_params
        ).to(self.device)

        return model
    
    def _build_optimizer(self, optimizer):
        opt_params = {}
        for k,v in self.params.items():
            if k in ['lr', 'eps', 'weight_decay', 'amsgrad']:
                opt_params[k] = v

        return TorchOptimizer.get_optimizer(
            optimizer=optimizer,
            model_params=self.model.parameters(),
            **opt_params
        )

    def fit(self, train_loader, valid_loader, epochs):
        # TODO: Add callbacks

        # Store loss for each epoch
        all_train_evals = []
        all_valid_evals = []
        for epoch in tqdm(range(epochs)):
            self._train(train_loader)

            train_eval = self._eval(train_loader)
            valid_eval = self._eval(valid_loader)
            all_train_evals.append(train_eval)
            all_valid_evals.append(valid_eval)

        return all_train_evals, all_valid_evals
    
    def _train(self, data_loader):
        self.model.train()
        for data in data_loader:
            self.optimizer.zero_grad()

            inputs = data["x"].to(self.device)
            targets = data["y"].to(self.device)
            outputs = self.model(inputs)
            loss = self.scoring(targets, outputs)

            loss.backward()
            self.optimizer.step()
    
    def _eval(self, data_loader):
        self.model.eval()
        final_loss = 0
        for data in data_loader:
            inputs = data["x"].to(self.device)
            targets = data["y"].to(self.device)
            outputs = self.model(inputs)
            loss = self.scoring(targets, outputs)

            final_loss += loss.item()
        
        return final_loss / len(data_loader)
