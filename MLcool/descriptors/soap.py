#!/usr/bin/env python3
from dscribe.descriptors import SOAP as dscribeSOAP
from ase import Atoms
import numpy as np
import copy
from ..utils.general_tools import sort_and_join

def make_subsystem(symbols, positions):
    return Atoms(symbols, positions)

class SOAP(object):
    def __init__(self, symbols, intermolecular = False):
        self.symbols        = symbols
        self.intermolecular = intermolecular

        self._build_labels()

    def _build_labels(self):
        labels = set()

        for at1 in self.symbols:
            for at2 in self.symbols:

                atom_pair = sort_and_join([at1, at2])
                labels.add(atom_pair)


        labels = list(labels)
        labels.sort()

        labels_dict = dict()
        for ind, key in enumerate(labels):
            if self.intermolecular:
                k = key + '_{}'
                labels_dict[k.format('intra')] = (0,ind)
                labels_dict[k.format('inter')] = (1,ind)

            else:
                labels_dict[key] = ind


        self.n_components   = len(labels)
        self.labels         = labels
        self.labels_dict    = labels_dict


        return self.labels


    def get(self, molecule, connectivity = None, nmax=3, lmax=2, rcut =100, sigma=1):

        with_inter  = self.intermolecular

        bag         = [[] for _ in range(self.n_components)]
        values      = [copy.deepcopy(bag) for _ in range(2)] if with_inter else bag

        syst = molecule.syst


        for i, atom_i in enumerate(syst):
            symb_i = atom_i.symbol
            pos_i  = atom_i.position
            conn_i = connectivity[i] if with_inter else None

            for j, atom_j in enumerate(syst):
                symb_j = atom_j.symbol
                pos_j  = atom_j.position

                if j <= i: continue

                key = sort_and_join([symb_i, symb_j])

                if with_inter:
                    interaction = 'intra' if conn_i[j] else 'inter'
                    key         = f'{key}_{interaction}'

                try:
                    idx     = self.labels_dict[key]

                except:
                    print('EEEEP')
                    continue


                sub_ij  = make_subsystem([symb_i, symb_j], [pos_i, pos_j])
                soap = dscribeSOAP(
                    species = [symb_i,symb_j],
                    periodic = False,
                    nmax    = nmax,
                    lmax    = lmax,
                    sigma   = sigma,
                    rcut    = rcut,
                    average = 'inner'
                )

                p_ij = soap.create(sub_ij, [0,1])

                if with_inter:
                    values[idx[0]][idx[1]].append(p_ij)

                else:
                    values[idx].append(p_ij)


        all_values = [[],[]] if with_inter else []
        if with_inter:
            for key in self.labels_dict:
                idx                     = self.labels_dict[key]
                v_key                   = values[idx[0]][idx[1]]
                v_key                   = [] if len(v_key) == 0 else np.sum(np.array(v_key), axis = 0)



                all_values[idx[0]]      = np.concatenate((all_values[idx[0]],v_key))

        else:
            for key in self.labels_dict:
                idx         = self.labels_dict[key]
                v_key       = values[idx]
                v_key       = np.sum(np.array(v_key), axis = 0)

                all_values  = np.concatenate((all_values,v_key))


        values = np.array(all_values)

        return self.labels, values




