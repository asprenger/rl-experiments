import os
import re
import tensorflow as tf
import numpy as np

def explained_variance(ypred, y):
    '''
    Computes fraction of variance that ypred explains about y.
    Returns 1 - Var[y-ypred] / Var[y]
    interpretation:
        ev=0  =>  might as well have predicted zero
        ev=1  =>  perfect prediction
        ev<0  =>  worse than just predicting zero
    '''
    assert y.ndim == 1 and ypred.ndim == 1
    vary = np.var(y)
    return np.nan if vary==0 else 1 - np.var(y - ypred) / vary
    

def last_vec_norm_path(model_path):
    if not os.path.exists(model_path):
        return None
    files = os.listdir(model_path)
    files = [file for file in files if file.startswith('vec_normalize-')]
    if len(files) == 0:
        return None
    epochs = [[i, int(re.search('vec_normalize-(.*).pickle', f).group(1))] for i,f in enumerate(files)]
    epochs.sort(key=lambda x: -x[1])
    last_epoch_idx = epochs[0][0]
    return os.path.join(model_path, files[last_epoch_idx])    