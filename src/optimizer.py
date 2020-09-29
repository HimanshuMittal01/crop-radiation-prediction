from torch import optim

class TorchOptimizer:
    @staticmethod
    def get_optimizer(optimizer, model_params, *args, **kwargs):
        res_optmizer = None
        if optimizer=='adam':
            res_optmizer =  optim.Adam(
                params=model_params,
                **kwargs
            )
        
        return res_optmizer