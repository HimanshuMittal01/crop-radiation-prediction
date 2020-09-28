import json
import argparse
import numpy as np
from src.data_utils import prepare_data
from src.models import StandaloneModel
from src.runner import Runner
from src.scoring import SklearnScorer

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help='path/to/input/config.json')
    args = parser.parse_args()

    config = None
    with open(args.config, 'r') as f:
        config = json.loads(f.read())
    
    # Get number of folds
    num_folds = config.get('num_folds', 1)

    # Automatically handles any file format
    data = prepare_data(
        filepath=config['input_filepath'],
        sheetname=config.get('sheet_name', None),
        delimiter=config.get('delimiter', None),
        num_folds=num_folds
    )
    
    # Initialize runner
    runner = Runner(
        data=data,
        feature_cols=config.get('feature_cols', None),
        target_cols=config.get('target_cols', None),
        batch_size=config.get('batch_size', 8),
        shuffle=config.get('shuffle', True),
        num_workers=config.get('num_workers', 0)
    )

    all_train_scores = []
    all_valid_scores = []
    for fold in range(num_folds):
        # Create new model for each fold
        model = StandaloneModel(
            algorithm=config.get('algorithm', 'random_forest'),
            problem_type=config.get('problem_type'),
            scoring=SklearnScorer.get_scoring_fn(config.get("scoring", None)),
            max_depth=10,
            random_state=42
        )

        # Run experiment on the model
        train_scores, valid_scores = runner.run_training(
            model=model,
            fold=fold,
            return_scores='best',
            direction=config.get('opt_direction')
        )

        print(f"#{fold}: Train: {train_scores}, Valid: {valid_scores}")

        all_train_scores.append(train_scores)
        all_valid_scores.append(valid_scores)

    # TODO: Handle case when return scores is all
    avg_train_score = np.mean(all_train_scores)
    avg_valid_score = np.mean(all_valid_scores)

    print("\n")
    print(f"Average training score for {num_folds} folds:", avg_train_score)
    print(f"Average validation score for {num_folds} folds:", avg_valid_score)