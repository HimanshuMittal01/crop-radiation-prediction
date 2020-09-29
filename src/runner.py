import torch
import numpy as np
import pandas as pd
from .dataloader import RadiationDataset
from .data_utils import define_data

class Runner:
    def __init__(
        self,
        data :pd.DataFrame,
        feature_cols : list,
        target_cols : list,
        batch_size :int,
        shuffle :bool,
        num_workers :int,
        epochs :int,
        mode :str
        ):
        """Runner for performing experiments
        """
        self.data = data
        self.feature_cols = feature_cols
        self.target_cols = target_cols
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.epochs = epochs
        self.mode = mode

    def get_algo_mode(self):
        return self.mode

    def run_training(self, model, fold, return_scores='best', direction='minimize'):
        """

        Parameters
        --------------------------------------------
        model : Model to train
        fold  : Which fold of dataset to validate
        return_scores : How to return scores
        direction : How to select best scores
        --------------------------------------------

        return_scores can take values: ['best','all']
        direction can take values: ['minimize','maximize']
        """
        train_data = self.data[self.data.kfold != fold].reset_index(drop=True)
        val_data   = self.data[self.data.kfold == fold].reset_index(drop=True)

        # Get train and validation features and target
        train_features, train_target = define_data(
            df = train_data,
            feature_cols=self.feature_cols,
            target_cols=self.target_cols
        )
        val_features, val_target = define_data(
            df = val_data,
            feature_cols=self.feature_cols,
            target_cols=self.target_cols
        )

        del train_data, val_data

        train_loader = torch.utils.data.DataLoader(
            dataset=RadiationDataset(
                features = train_features,
                target = train_target,
                mode = self.mode
            ),
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers
        )
        valid_loader = torch.utils.data.DataLoader(
            dataset=RadiationDataset(
                features = val_features,
                target = val_target,
                mode = self.mode
            ),
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers
        )

        del train_features, train_target, val_features, val_target

        # Fit the model
        train_scores, valid_scores = model.fit(train_loader, valid_loader, self.epochs)

        # Return scores
        if return_scores=='all':
            return train_scores, valid_scores
        elif return_scores=='best':
            if direction=='minimize':
                return np.min(train_scores), np.min(valid_scores)
            elif direction=='maximize':
                return np.max(train_scores), np.max(valid_scores)
