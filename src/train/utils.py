import yaml
from argparse import Namespace
from ..utils.path import WORKDIR

def get_args(rname=None, fname=None, tname=None):
    if rname is not None:
        with open(f"{WORKDIR}/cplm/reinforce/results/{rname}/args.yaml") as f:
            rargs = Namespace(**yaml.safe_load(f))
        fname = rargs.finetune_name
    else:
        rargs = None
    if fname is not None:
        with open(f"{WORKDIR}/cplm/finetune/results/{fname}/args.yaml") as f:
            fargs = Namespace(**yaml.safe_load(f))
        tname = fargs.pretrain_name
    else:
        fargs = None
    if tname is not None:
        with open(f"{WORKDIR}/cplm/training/results/{tname}/args.yaml") as f:
            targs = Namespace(**yaml.safe_load(f))
    else:
        targs = None
    return targs, fargs, rargs