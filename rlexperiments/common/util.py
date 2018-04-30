import os
import random
import datetime
import numpy as np

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def ortho_init(scale=1.0):
    def _ortho_init(shape, dtype, partition_info=None):
        #lasagne ortho init for tf
        shape = tuple(shape)
        if len(shape) == 2:
            flat_shape = shape
        elif len(shape) == 4: # assumes NHWC
            flat_shape = (np.prod(shape[:-1]), shape[-1])
        else:
            raise NotImplementedError
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v # pick the one with the correct shape
        q = q.reshape(shape)
        return (scale * q[:shape[0], :shape[1]]).astype(np.float32)
    return _ortho_init

def standardize_features(x):
    '''Returns the values of x having zero-mean and unit-variance.'''
    return (x - x.mean()) / (x.std() + 1e-8)
    
def ts_rand():
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    random_num = random.randint(1e6, 1e7-1)
    return '%s_%d' % (ts, random_num)

def ts():
    return datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
