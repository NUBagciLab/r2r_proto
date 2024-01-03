import importlib
from os.path import join


class Configs:
    def __init__(self, **kwargs) -> None:
        for k,v in kwargs.items():
            setattr(self, k, v)


def parse(exp_name, result_dir='_results'):
    opt = importlib.import_module('exps.'+exp_name).get()
    
    dset_name, expid = exp_name.split('.')

    opt['outdir'] = join(result_dir, dset_name, expid, 'out')
    opt['modeldir'] = join(result_dir, dset_name, expid, 'model')
    opt['logdir'] = join(result_dir, dset_name, expid, 'log')
    opt['slurmdir'] = join(result_dir, dset_name, expid, 'slurm')

    #opt = Configs(**opt)
    return opt