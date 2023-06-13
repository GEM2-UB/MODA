#! /usr/bin/env python3
from ..utils.general_tools import sort_and_join
import numpy as np
from functools import reduce


class MODA(object):
    def __init__(self, symbols, principal, azimuthal, intermolecular = False):
        self.symbols        = symbols
        self.principal      = principal
        self.azimuthal      = azimuthal
        self.intermolecular = intermolecular

        self._build_labels()


    def _build_labels(self):
        labels  = set()
        for at1 in self.symbols:
            for at2 in self.symbols:

                atom_pair = sort_and_join([at1, at2])

                for n1 in self.principal:
                    for n2 in self.principal:

                        for l in range(0, np.min([n1,n2])):
                            if l in self.azimuthal:
                                labels.add(f'{atom_pair}_{n1}{n2}_{l}')


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


    def set_attribute(self, attr_name, attr_value):
        if attr_name in ['symbols', 'principal', 'azimuthal', 'intermolecular']:
            if attr_name == 'symbols'           : self.symbols          = attr_value
            if attr_name == 'principal'         : self.principal        = attr_value
            if attr_name == 'azimuthal'         : self.azimuthal        = attr_value
            if attr_name == 'intermolecular'    : self.intermolecular   = attr_value

            self._build_labels()

        else:
            print('Unknown MODA Attribute. Exit.')


    def get(self, molecule, orbital_numbers, connectivity = None):

        target_coef = molecule.C[:,orbital_numbers]
        S           = molecule.get_overlap()

        n_basis     = molecule.basis_summary.shape[0]
        basis_range = range(n_basis)

        with_inter  = self.intermolecular
        values      = np.zeros((2, self.n_components)) if with_inter else np.zeros(self.n_components)

        for b_i in basis_range:
            ind_i, symb_i, lab_i, n_i, l_i, m_i = molecule.basis_summary[b_i]
            c_i                                 = target_coef[b_i]
            index_i                             = connectivity[ind_i] if with_inter else None

            for b_j in basis_range:

                ind_j, symb_j, lab_j, n_j, l_j, m_j = molecule.basis_summary[b_j]
                c_j                                 = target_coef[b_j]


                if l_i != l_j or ind_i == ind_j:    continue
                if b_j <= b_i:                      continue

                pair    = sort_and_join([symb_i, symb_j])
                pr      = symb_i + symb_j
                key     = f'{pair}_{n_i}{n_j}_{l_i}' if pair == pr else f'{pair}_{n_j}{n_i}_{l_i}'

                if with_inter:
                    interaction_type     = 'intra' if index_i[ind_j] else 'inter'
                    key                 += '_{}'.format(interaction_type)

                try:
                    idx = self.labels_dict[key]

                except:
                    print('Warning in method get() of MODA: index not found in labels.')
                    print('Dismiss and continue.')
                    continue


                S_ij = S[b_i,b_j]
                d_ij = np.multiply(c_i, c_j)
                p_ij = np.sum(d_ij)*S_ij


                if with_inter:
                    values[idx[0], idx[1]]  += p_ij

                else:
                    values[idx]             += p_ij


        return self.labels, values


