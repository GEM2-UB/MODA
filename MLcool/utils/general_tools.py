#!/usr/bin/env python3
import numpy as np
from itertools import combinations

from ..learning.kernels import Normalize

def sort_and_join(lst):
    lst.sort()

    return ''.join(lst)


def charge_dataset(path, step = 1):
    descriptor      = np.load(path)
    feat, target    = descriptor['features'], descriptor['couplings']

    return feat[::step], target[::step]


def concatenate_and_shuffle(ds):
    first = True
    for i, this_ds in enumerate(ds):
        feat, Js    = this_ds

        if first:
            f, j    = feat, Js
            first   = False
        else:
            f       = np.concatenate((f,feat))
            j       = np.concatenate((j,Js))


    # Shuffle:
    l = len(f)
    index = np.arange(l)
    np.random.shuffle(index)
    f = f[index]
    j = j[index]

    return f, j


def split_in_n_groups(n_samples, n_splits):
    idx = np.array(list(range(n_samples)))

    return np.array_split(idx, n_splits, axis = 0)


def leave_p_groups_out(n_groups, p_out):
    n_in    = n_groups - p_out

    r       = range(n_groups)
    tot     = np.array(r)
    all_in  = [list(comb) for comb in combinations(r, n_in)]

    all_out = []
    for c in all_in:
        out_ind = [n not in c for n in r]
        all_out.append(list(tot[out_ind]))


    splits = list(zip(all_in, all_out))

    return splits


def get_tr_ts_splits(len_dataset, n_groups, groups_out):
    splits = split_in_n_groups(len_dataset, n_groups)
    groups = leave_p_groups_out(n_groups, groups_out)

    training, testing = [], []

    for gr in groups:
        tr, ts = gr
        
        this_tr, this_ts = [], []
        for idx, split in enumerate(splits):
            if idx in tr:
                this_tr += list(split)

            elif idx in ts:
                this_ts += list(split)


        training.append(this_tr)
        testing.append(this_ts)


    return training, testing


def shuffle_and_split(ds, n_groups, with_normalization = False):
    # Concatenate:
    first = True
    for i, this_ds in enumerate(ds):
        feat, Js    = this_ds

        if first:
            f, j    = feat, Js
            first   = False
        else:
            f       = np.concatenate((f,feat))
            j       = np.concatenate((j,Js))


    # Normalize if requested:
    if with_normalization:
        nr = Normalize()
        f  = nr.fit_transform(f)


    # Shuffle:
    l = len(f)
    index = np.arange(l)
    np.random.shuffle(index)
    f = f[index]
    j = j[index]

    # Split:
    f_splits = np.array_split(f, n_splits, axis = 0)
    j_splits = np.array_split(j, n_splits, axis = 0)

    return [[f_splits[i], j_splits[i]] for i in range(n_splits)]



def get_monomers(syst, thr = 2.0, sort_connectivity = True, verbose = False):
    dist            = syst.get_all_distances(mic=True)
    A, D, L         = get_ADL(dist, thr)
    vals, vects     = np.linalg.eigh(L)

    zeros           = np.argwhere(np.abs(vals) < 1.0e-5).T[0]
    target_vects    = np.abs(vects[:,zeros])
    
    if verbose: print(f'Number of monomers: {len(zeros)}')

    l               = len(syst)
    monomers        = []
    connectivity    = np.zeros((l,l), dtype=bool)

    for i, v in enumerate(target_vects.T):
        x       = np.max(v)
        n_edges = np.around(1/x**2)
        v_bool  = (v > x/2)

        monomers.append(syst[v_bool].copy())
        
        subgraph = [i for i, v in enumerate(v_bool) if v]
        for tr1 in subgraph:
            for tr2 in subgraph:
                connectivity[tr1, tr2] = True


        if verbose:
            print(f'Monomer {i:3} --> {np.int(n_edges):3} atom(s)')


    if sort_connectivity:
        for i, m in enumerate(monomers):
            if i == 0:
                om = m.copy()
            else:
                om += m.copy()

        _, connectivity = get_monomers(om, thr, False, False)


    return monomers, connectivity


def get_ADL(distances, thr):
    A  = 1*np.greater(thr, distances)
    A -= np.diag(np.diag(A))

    D  = np.diag(np.sum(A, axis = 1))
    L  = D - A

    return A, D, L


