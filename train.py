import json
import argparse
import numpy as np
import optuna
from functools import partial
from src.data_utils import prepare_data
from src.models import StandaloneModel
from src.runner import Runner
from src.scoring import SklearnScorer
from src.optimizer import MLHyperOptmizer

def objective_function(trial, params, config, runner):
    all_valid_scores = []
    for fold in range(config.get('num_folds')):
        # Creating hyper parameters grid in optuna space
        params = MLHyperOptmizer.optuna_space(trial, params)

        # Create new model for each fold
        model = None
        if runner.get_algo_mode()=="ml_standalone":
            model = StandaloneModel(
                algorithm=config.get('algorithm'),
                problem_type=config.get('problem_type'),
                scoring=SklearnScorer.get_scoring_fn(config.get("scoring", None)),
                random_state=config.get('random_state'),
                **params
            )

        # Run experiment on the model
        train_scores, valid_scores = runner.run_training(
            model=model,
            fold=fold,
            return_scores='best',
            direction=config.get('opt_direction')
        )

        all_valid_scores.append(valid_scores)

    avg_valid_score = np.mean(all_valid_scores)
    
    return avg_valid_score

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help='path/to/input/config.json')
    args = parser.parse_args()

    config = None
    with open(args.config, 'r') as f:
        config = json.loads(f.read())
    
    # Automatically handles any file format
    data = prepare_data(
        filepath=config['input_filepath'],
        sheetname=config.get('sheet_name', None),
        delimiter=config.get('delimiter', None),
        num_folds=config.get('num_folds'),
        random_state=config.get('random_state')
    )
    
    # Initialize runner
    runner = Runner(
        data=data,
        feature_cols=config.get('feature_cols', None),
        target_cols=config.get('target_cols', None),
        batch_size=config.get('batch_size', 8),
        shuffle=config.get('shuffle', True),
        num_workers=config.get('num_workers', 0),
        mode=config.get('algo_mode')
    )

    params = MLHyperOptmizer.get_params(config.get('hyper_param_space'))

    optimize_fn = partial(
        objective_function,
        params=params,
        config=config,
        runner=runner
    )

    study = optuna.create_study(direction=config.get('opt_direction'))
    study.optimize(optimize_fn, n_trials=config.get('num_opt_trials'))

    print(study.best_params)