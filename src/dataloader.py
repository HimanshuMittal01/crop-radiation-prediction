import numpy as np
import torch

class RadiationDataset:
    """
    Features
    ------------------
    # Tabular dataset
    # Single target variable'
    # Regression output
    # Works for both machine and deep learning frameworks
    """
    DEFAULT_MODE = "ml"
    AVAILABLE_MODES = ["ml","torch_dnn","torch_lstm"]

    def __init__(self, features, target, mode="ml"):
        self.mode = mode.lower()
        self.features = features
        self.target = target

        self._fix_features()
        self._fix_target()

        assert(self.features.shape[0]==self.target.shape[0]), "Inconsistent number of samples in features and target"
    
    def __len__(self):
        return self.features.shape[0]
    
    def __getitem__(self, index):
        if self.mode=="ml":
            return {
                "x" : self.features.iloc[index, :],
                "y" : self.target.iloc[index, :]
            }
        elif self.mode=="dl":
            return {
                "x" : torch.tensor(self.features.iloc[index, :], dtype=torch.float),
                "y" : torch.tensor(self.target.iloc[index, :], dtype=torch.float)
            }
    
    def _fix_features(self):
        if not isinstance(self.features, np.ndarray):
            self.features = np.array(self.features)

        assert(self.features.ndim==2), "Features dimensions is not compatible"
    
    def _fix_target(self):
        if not isinstance(self.target, np.ndarray):
            self.target = np.array(self.target)
        
        self.target.reshape(-1,1)
    
    def _fix_mode(self):
        if self.mode not in RadiationDataset.AVAILABLE_MODES or self.mode is None:
            self.mode = RadiationDataset.DEFAULT_MODE
    
    def get_full_dataset(self):
        """
        Exclusive function for loading whole dataset at once
        Not recommended to use if dataset is large
        Instead use partial_fit() in sklearn model to train in batches
        """

        # return X, y
        return self.features, self.target