#!/usr/bin/env python3
import numpy as np
import copy
from ..utils.general_tools import sort_and_join
from functools import reduce


def parse_and_order(bob):
    bob = np.array([-1*np.sort(-1*np.array(bag)) for bag in bob], dtype=object)

    return reduce(lambda a, b: np.concatenate((a,b)), bob)


class BOB(object):
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


    def get(self, molecule, connectivity = None):

        with_inter  = self.intermolecular

        bag         = [[] for _ in range(self.n_components)]
        values      = [copy.deepcopy(bag) for _ in range(2)] if with_inter else bag

        syst        = molecule.syst

        for ind_i, at_i in enumerate(syst):
            symb_i      = at_i.symbol
            charge_i    = at_i.number
            index_i     = connectivity[ind_i] if with_inter else None


            for ind_j, at_j in enumerate(syst):
                symb_j      = at_j.symbol
                charge_j    = at_j.number

                if ind_j <= ind_i: continue

                key         = sort_and_join([symb_i, symb_j])

                if with_inter:
                    interaction = 'intra' if index_i[ind_j] else 'inter'
                    key         = f'{key}_{interaction}'

                try:
                    idx     = self.labels_dict[key]

                except:
                    continue


                dist_ij     = syst.get_distance(ind_i, ind_j, mic = True, vector = False)
                p_ij        = charge_i*charge_j/dist_ij
                #p_ij        = 1/dist_ij


                if with_inter:
                    values[idx[0]][idx[1]].append(p_ij)

                else:
                    values[idx].append(p_ij)



        if with_inter:
            intra           = parse_and_order(values[0])
            inter           = parse_and_order(values[1])


            lv              = np.max([len(intra), len(inter)])
            v               = np.zeros((2,lv))

            v[0,:len(intra)] = intra
            v[1,:len(inter)] = inter
            values = v


        else:
            values = parse_and_order(values)


        return self.labels, values

