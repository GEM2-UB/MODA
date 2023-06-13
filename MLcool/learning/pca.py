#!/usr/bin/env python3

import numpy as np
from .kernels import SimpleKernel


class KernelSampleSelection(object):
    def __init__(self, n_comp, g, ktype = 'rbf'):
        if n_comp < 1:
            self.n_comp         = None
            self.v_explained    = n_comp

        else:
            self.n_comp         = n_comp
            self.v_explained    = None


        self.g      = g
        self.ktype  = ktype
        self.k_func = SimpleKernel(self.g, self.ktype)
        self.kernel = None


    def get(self, f1, n_jobs = 1):
        self.kernel = self.k_func.get(f1, n_jobs =  n_jobs)

        # Precedure to center the kernel
        l           = len(self.kernel)
        O           = np.eye(l) - np.ones((l,l))/l
        self.kernel = O@self.kernel@O

        # Perform the eigendecomposition (KernelPCA in the sample's space)
        D, V        = np.linalg.eigh(self.kernel)

        # Reverse order and normalize eigenvalues to (%) of contribution
        s           = np.sum(D)
        idx         = np.argsort(-D)
        D           = D[idx]/s
        V           = V[:,idx]

        if self.n_comp is not None:
            self.selected       = [ d_i < self.n_comp for d_i, _ in enumerate(D) ]
            self.v_explained    = np.sum(D[self.selected])

        else:
            self.selected       = []
            cummulative         = 0

            d_i = 0
            while cummulative < self.v_explained:
                cummulative += D[d_i]
                d_i         += 1

            self.n_comp      = d_i
            self.v_explained = cummulative
            self.selected    = [ d_i < self.n_comp for d_i, _ in enumerate(D) ]


        self.evals          = D
        self.selected_evals = D[self.selected]
        self.evecs          = V
        self.selected_evecs = V[:, self.selected]

        return self._get_weights()
        

    def _get_weights(self):
        vecs2 = np.power(self.selected_evecs,2)
        prod  = [self.selected_evals[i]*u_i for i, u_i in enumerate(vecs2.T)]

        weight = np.sqrt(np.sum(prod, axis = 0))

        return weight

    def caca(self, f1, thr = 0.98, n_jobs = -1):
        self.kernel = self.k_func.get(f1, n_jobs =  n_jobs)
        l           = len(self.kernel)
        removed     = set()

        import matplotlib.pyplot as plt
        plt.imshow(self.kernel)
        plt.show()


        for sel in range(l):
            if sel not in removed:

                equal = [i for i in range(sel+1,l) if self.kernel[sel,i] > thr]
                print(sel, equal)
                for e in equal:
                    removed.add(e)


        chosen = [i for i in range(l) if i not in removed]


        return chosen





class CustomPCA(object):
    def __init__(self, n_comp):
        if n_comp < 1:
            self.n_comp         = None
            self.v_explained    = n_comp

        else:
            self.n_comp         = n_comp
            self.v_explained    = None

        self.id_fitted = False


    def fit(self, features):
        C       = self._get_covariance(features)
        D, V    = np.linalg.eigh(C)
        s       = np.sum(D)

        idx     = np.argsort(-D)
        D       = D[idx]/s
        V       = V[:,idx]

        if self.n_comp is not None:
            self.selected       = [ d_i < self.n_comp for d_i, _ in enumerate(D) ]
            self.v_explained    = np.sum(D[self.selected])

        else:
            self.selected       = []
            cummulative         = 0

            d_i = 0
            while cummulative < self.v_explained:
                cummulative += D[d_i]
                d_i         += 1

            self.n_comp      = d_i
            self.v_explained = cummulative
            self.selected    = [ d_i < self.n_comp for d_i, _ in enumerate(D) ]


        self.evals          = D
        self.selected_evals = D[self.selected]
        self.evecs          = V
        self.selected_evecs = V[:, self.selected]


        self.is_fitted = True


    def transform(self, features, full = False):
        if self.is_fitted:
            V = self.evecs if full else self.selected_evecs
            T = features@V

        else:
            print('Fit the PCA before transforming data')
            print('Exit.')
            exit()

        return T


    def fit_transform(self, features, full = False):
        self.fit(features)

        return self.transform(features, full)


    def _get_covariance(self, features):
        B = features - np.mean(features, axis = 0)
        C = B.T@B
        C /= len(C) - 1

        

        return C


    def __repr__(self):
        ret_string = ''

        l       = len(self.evals)
        string  = 'i = {:5} (v = {:+8.2f}%)' + '{:+8.4f} '*l + '({})\n'
        for i, (d, v) in enumerate(zip(self.evals, self.evecs.T)):
            taken = 'TAKEN' if self.selected[i] else 'NOT TAKEN'

            ret_string += string.format(i, 100*d, *v, taken)

        return ret_string


