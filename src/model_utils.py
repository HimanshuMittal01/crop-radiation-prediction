from .models import (
    StandaloneModel,
    DNNRegression
)
from .scoring import SklearnScorer, TorchScorer

def build_model(config, params):
    model = None
    if config.get('algo_mode')=="ml_standalone":
        model = StandaloneModel(
            algorithm=config.get('algorithm'),
            problem_type=config.get('problem_type'),
            scoring=SklearnScorer.get_scoring_fn(config.get("scoring", None)),
            random_state=config.get('random_state'),
            **params
        )
    elif config.get('algo_mode')=="torch_dnn":
        model = DNNRegression(
            problem_type=config.get('problem_type'),
            scoring=TorchScorer.get_scoring_fn(config.get('scoring')),
            optimizer=config.get('optimizer'),
            num_features=config.get('num_features'),
            num_targets=config.get('num_targets'),
            hidden_layers=config.get('hidden_layers'),
            dropout_rate=config.get('dropout_rate'),
            device=config.get('device')
        )

    return model
