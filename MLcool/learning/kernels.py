#!/usr/bin/env python3

from .optimized_kernels import get_rbf, get_rbf_asym
from .optimized_kernels import get_lap, get_lap_asym
from .optimized_kernels import get_normalization, get_normalization_attributes


from joblib import Parallel, delayed, cpu_count
import numpy as np

class SimpleKernel(object):
    def __init__(self, g, ktype = 'rbf'):
        self.g      = g
        self.ktype  = ktype

        if ktype not in ['rbf', 'laplacian']:
            print('Kernel must be either "rbf" or "laplacian" ')
            print('Exit.')
            exit()


    def get(self, f1, f2 = None, n_jobs = 1):
        if n_jobs == 1:
            if f2 is None:
                if self.ktype == 'rbf':
                    return get_rbf(f1, self.g)

                elif self.ktype == 'laplacian':
                    return get_lap(f1, self.g)


            if self.ktype == 'rbf':
                return get_rbf_asym(f2, f1, self.g)

            elif self.ktype == 'laplacian':
                return get_lap_asym(f2, f1, self.g)


        else:
            n_jobs          = cpu_count() if n_jobs == -1 else n_jobs

            sym             = f2 is None
            f1_splits       = np.array_split(f1, n_jobs, axis = 0)
            f2_splits       = f1_splits if sym else np.array_split(f2, n_jobs, axis = 0)

            parallel_jobs   = []

            for i, s1 in enumerate(f1_splits):
                for j, s2 in enumerate(f2_splits):
                    idx = (i,j)
                    d   = delayed(sub_get_simple)
                    d   = d(s1, s2, self, sym, idx)
                    parallel_jobs.append(d)


            r = Parallel(n_jobs = n_jobs)(parallel_jobs)

            try:
                results = np.reshape(r, (n_jobs, n_jobs), dtype = object)

            except:
                results = np.zeros((n_jobs, n_jobs), dtype = object)

                v_i = 0
                for i in range(n_jobs):
                    for j in range(n_jobs):
                        results[i,j] = r[v_i]
                        v_i += 1

            del r

            kernel = np.concatenate(
                    tuple([
                        np.concatenate(
                            tuple([
                                results[i,j] for j in range(n_jobs)
                                ]),
                            axis = 0
                            )
                        for i in range(n_jobs)]),
                    axis = 1
                    )

        
            return kernel


def sub_get_simple(s1, s2, obj, sym, idx):
    i, j = idx
    cond = i == j and sym

    if obj.ktype == 'rbf':
        return get_rbf(s1, obj.g) if cond else get_rbf_asym(s2, s1, obj.g)

    elif obj.ktype == 'laplacian':
        return get_lap(s1, obj.g) if cond else get_lap_asym(s2, s1, obj.g)


#def sub_get_simple(s1, s2, obj, sym, idx):
#    i, j = idx
#    if i == j and sym:
#        k = get_rbf(s1, obj.g)
#
#    else:
#        k = get_rbf_asym(s2, s1, obj.g)
#
#    return k


class Normalize(object):
    def __init__(self):
        self.axis_max = None


    def fit(self, feat):
        self.axis_max = get_normalization_attributes(feat)


    def transform(self, feat):
        if self.axis_max is None:
            print('Call method "fit" first in Normalize() instance.')
            exit()

        return get_normalization(feat, self.axis_max)


    def fit_transform(self, feat):
        self.fit(feat)

        return self.transform(feat)



