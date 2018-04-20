import os
import sys
import multiprocessing
import numpy as np
import tensorflow as tf

def create_session(cuda_visible_devices='0', gpu_memory_fraction=0.5):
    ncpu = multiprocessing.cpu_count()
    if sys.platform == 'darwin': ncpu //= 2
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=ncpu,
                            inter_op_parallelism_threads=ncpu)
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = gpu_memory_fraction
    return tf.Session(config=config)

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
    