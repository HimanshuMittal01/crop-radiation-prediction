class HyperRange:
    def __init__(self, name, low, high, dtype, step=None, log=False):
        self.name=name
        self.low=low
        self.high=high
        self.dtype=dtype
        self.step=step
        self.log=log

        self._fix_step()
        self._fix_end()

    def _fix_step(self):
        if self.step is not None:
            return
        if self.dtype=='int':
            self.step = 1
    
    def _fix_end(self):
        if self.step is None:
            return
        self.high = self.low + self.step * ((self.high - self.low)//self.step)

class HyperCategorical:
    def __init__(self, name, choices):
        self.name=name
        self.choices=choices

class MLHyperOptmizer:
    """
    """
    rf_reg_small_1 = {
        HyperRange('n_estimators', 100, 1500, 'int', 100),
        HyperRange('max_depth', 1, 31, 'int', 2),
        HyperCategorical('criterion', ['mse', 'mae']),
        HyperRange('min_samples_split', 2, 15, 'int', 1)
    }
    gb_reg_small_1 = {
        HyperRange('n_estimators', 100, 1500, 'int', 100),
        HyperRange('learning_rate', 0.001, 0.1, 'float'),
        HyperRange('subsample', 0.8, 1, 'float'),
        HyperRange('min_samples_split', 2, 15, 'int', 1)
    }
    ridge_reg_small_1 = {
        HyperRange('alpha', 0.1, 20, 'float'),
        HyperCategorical('solver', ['auto','sparse_cg','sag'])
    }
    @staticmethod
    def get_super_space():
        super_space = {
            "rf_reg_small_1": MLHyperOptmizer.rf_reg_small_1,
            "gb_reg_small_1": MLHyperOptmizer.gb_reg_small_1,
            "ridge_reg_small_1": MLHyperOptmizer.ridge_reg_small_1
        }

        return super_space
    
    @staticmethod
    def get_params(params):
        super_space = MLHyperOptmizer.get_super_space()
        if params not in super_space:
            raise Exception("Not available/implemented")

        return super_space[params]
    
    @staticmethod
    def optuna_space(trial, params):
        optuna_space = {}

        # TODO: Add all types of variables
        for hrange in  params:
            if isinstance(hrange, HyperRange):
                if hrange.dtype=='int':
                    optuna_space[hrange.name] = trial.suggest_int(name=hrange.name, low=hrange.low, high=hrange.high, step=hrange.step, log=hrange.log)
                elif hrange.dtype=='float':
                    optuna_space[hrange.name] = trial.suggest_float(name=hrange.name, low=hrange.low, high=hrange.high, step=hrange.step, log=hrange.log)
                else:
                    pass
            elif isinstance(hrange, HyperCategorical):
                optuna_space[hrange.name] = trial.suggest_categorical(hrange.name, hrange.choices)

        return optuna_space