#!/usr/bin/env python3

# pySCF:
# -------------------------------------------
from pyscf.gto import Mole
from pyscf.scf import rohf
from pyscf.tools.cubegen import orbital

# scipy and numpy:
# -------------------------------------------
import numpy as np
import scipy.linalg as LA
from scipy.linalg import fractional_matrix_power

# ASE:
# -------------------------------------------
from ase import Atoms

# Other:
# -------------------------------------------
from functools import reduce

from ..descriptors import MODA, SOAP, BOB


func2num = {
        's'         : (0,(0,0,0)),
        'px'        : (1,(1,0,0)),
        'py'        : (1,(0,1,0)),
        'pz'        : (1,(0,0,1)),
        'dxy'       : (2,(1,1,0)),
        'dxz'       : (2,(1,0,1)),
        'dyz'       : (2,(0,1,1)),
        'dz^2'      : (2,(0,0,2)),
        'dx2-y2'    : (2,(2,-2,0)),
        }


def func2index(func):
    n           = int(func[0])
    func_index  = func2num[func[1:]]

    return (n, *func_index)


class Molecule(object):
    __doc__ = r'''-----------------------------
    > Description: 
        base object to work in MLcool. Contains all the methods required 
        to extract the descripotrs of chemical system. It builds a pyscf.gto.Mole object 
        from the atributes provided in the constructor of the class.

    > Parameters:
        >> syst (ase.Atoms): 
            ASE atoms object containing positions anc chemical symbols.
        >> spin (int): 
            Number of unpaired electrons, 2S, of the system.
        >> charge (int):
            Charge of the chemical system.
        >> basis (str):
            Name of the Gaussian-type Orbital (GTO) basis set. 
            It have to be compatible basis format of pySCF.

    > Note: Contains wrapper methods to some descriptors:
        >> get_MODA()
        >> get_BOB()
        >> get_SOAP()

    > Example:
        from ase.build import molecule
        syst = molecule('H2O')
        mol  = Molecule(syst, 0, 0, 'sto3g')
        print(mol)

        occup, coeff = mol.solve_H()
        mol.get_orbital(mol.noccupied-1, cube_name='./cubet_test.cube') # HOMO

        labels, values = mol.get_MODA(
            orbital_numbers = mol.noccupied-1,
            intermolecular  = False,
            connectivity    = None,
            verbose         = True
        )

    '''
    def __init__(self, syst: Atoms, spin: int, charge: int, basis: str): 

        # Parameters of the constructor:
        # -------------------------------------------
        self.syst       = syst
        self.spin       = spin
        self.charge     = charge
        self.basis      = basis

        # Build up:
        # -------------------------------------------
        self._build_mole()
        self._make_basis_summary()

        self.natoms     = len(syst)
        self.noccupied  = self.get_noccupied()

        # Initialize some matrices:
        self.D, self.S, self.C, self.N              = None, None, None, None


    def _build_mole(self):
        symbols         = self.syst.get_chemical_symbols()
        positions       = self.syst.get_positions()
        coords          = [[atom, tuple(pos)] for atom, pos in zip(symbols,positions)]


        mole            = Mole()
        mole.atom       = coords
        mole.charge     = self.charge
        mole.spin       = self.spin
        mole.basis      = self.basis
        mole.verbose    = False
        mole.build()

        self.nbasis     = len(mole.ao_labels())
        self.mole       = mole


    def _make_basis_summary(self):
        labels = self.mole.ao_labels()

        parsed_labels = []
        for lab in labels:
            parsed_lab  = lab.split()

            atom_index  = int(parsed_lab[0])
            atom_number = parsed_lab[1]
            basis_name  = parsed_lab[2].strip()
            orb_type    = func2index(basis_name)

            parsed_labels.append([atom_index, atom_number, basis_name, *orb_type])

        self.basis_summary = np.array(parsed_labels, dtype=object)


    def get_noccupied(self):
        n_elec      = np.sum(self.mole.nelec)
        n_unpaired  = n_elec - 2*(n_elec//2)
        n_occupied  = n_elec//2 + n_unpaired

        #return n_occupied
        return self.mole.nelec[0]

    
    def get_overlap(self):
        if self.S is None:
            self.S = self.mole.intor('int1e_ovlp')

        return self.S


    def solve_H(self):
        self.D          = reduce(lambda a, b: a+b, rohf.init_guess_by_atom(self.mole))
        self.S          = self.get_overlap()

        S1m             = fractional_matrix_power(self.S, +0.5)
        S1m_inv         = fractional_matrix_power(self.S, -0.5)

        self.N, self.C  = LA.eigh(S1m@self.D@S1m)

        self.N          = self.N[::-1]
        self.C          = self.C[:,::-1]
        self.C          = S1m_inv@self.C

        return self.N, self.C


    def __add__(self, other):
        syst1, syst2        = self.syst, other.syst
        pos1, pos2          = syst1.get_positions(),        syst2.get_positions()
        symb1, symb2        = syst1.get_chemical_symbols(), syst2.get_chemical_symbols()

        spin1, spin2        = self.spin,    other.spin
        charge1, charge2    = self.charge,  other.charge
        basis1, basis2      = self.basis,   other.basis

        new_syst            = Atoms(symb1 + symb2, np.concatenate((pos1, pos2), axis = 0))

        try:
            assert basis1 == basis2

            new_spin    = spin1 + spin2
            new_charge  = charge1 + charge2
            new_basis   = basis1

        except:
            print('ERROR: Adding two Molecule() instances with different basis')
            print('is not supported. Exit.')
            exit()


        new_molecule    = Molecule(new_syst, new_spin, new_charge, new_basis)
        
        are_C_calc      = self.C is not None and other.C is not None

        if not are_C_calc:
            return new_molecule


        n1, n2  = self.nbasis, other.nbasis
        S12     = new_molecule.get_overlap()[:n1, n1:].T
        new_C   = []

        for c1, c2 in zip(self.C.T, other.C.T):
            k1      = c1.reshape(-1,+1)
            k2      = c2.reshape(+1,-1)
            ovlp    = np.matmul(k2, np.matmul(S12, k1))[0,0]

            normalization_pls   = 1/np.sqrt(2*(1+ovlp))
            normalization_mns   = 1/np.sqrt(2*(1-ovlp))

            new_pls_C           = normalization_pls*np.concatenate((+c1, +c2))
            new_mns_C           = normalization_mns*np.concatenate((+c1, -c2))

            if ovlp > 0.0E0:
                new_C.append(new_pls_C)
                new_C.append(new_mns_C)

            else:
                new_C.append(new_mns_C)
                new_C.append(new_pls_C)


        new_molecule.C = np.transpose(new_C)
        new_molecule.N = np.array([None for _ in range(n1+n2)], dtype = object)


        return new_molecule


    def get_orbital(self, orb_num, grid = (60,60,60), with_cube = True, with_summary = True, cube_name = 'orbital.cube'):
        if self.N is None or self.C is None or orb_num >= len(self.N) :
            print('Occupation and/or coefficients matrix not found.')
            print('Calculate the orbitals before.')
            return

        coef = self.C[:,orb_num]
        occu = self.N[orb_num]

        if with_summary:
            print_template = '{}_{}\t{}\t{:+.4e}'

            if occu is not None:
                print('Orbital number: {:5} (Occupation = {:14.5f})'.format(orb_num, occu))
            else:
                print('Orbital number: {:5}'.format(orb_num))


            print('-'*20)
            for label, c in zip(self.basis_summary, coef):
                print(print_template.format(label[1], label[0], label[2], c))

            print('-'*20 + '\n')


        if with_cube:
            print(f'Generating cube file "{cube_name}" ...')
            nx, ny, nz = grid
            orbital(self.mole, cube_name, coef, nx, ny, nz)
            print('DONE\n')

    
    def get_MODA(self, orbital_numbers, symbols = None, principal = None,
            azimuthal = None, intermolecular = None, connectivity = None, verbose = False):

        if symbols is None:         symbols         = set(self.syst.get_chemical_symbols())
        if principal is None:       principal       = set(self.basis_summary[:,3])
        if azimuthal is None:       azimuthal       = set(self.basis_summary[:,4])
        if intermolecular is None:  intermolecular  = False

        m               = MODA(symbols, principal, azimuthal, intermolecular)
        labels, values  = m.get(self, orbital_numbers, connectivity)


        if verbose:
            print('MODA representation:')
            print('-'*20)
            i = 0
            if not intermolecular:
                for l, v in zip(labels, values):
                    print('({}) {}\t\t{:+.5e}'.format(i, l, v))
                    i += 1

            else:
                for l, v in zip(labels, values.T):
                    print('({}) {}\t\t{:+.5e}\t{:+.5e}'.format(i, l, *v))
                    i += 1


        return labels, values


    def get_BOB(self, symbols = None, intermolecular = None, connectivity = None, verbose = False):
        if symbols is None:         symbols         = set(self.syst.get_chemical_symbols())
        if intermolecular is None:  intermolecular  = False

        b               = BOB(symbols, intermolecular)
        labels, values  = b.get(self, connectivity)


        if verbose:
            print('BOB representation:')
            print('-'*20)
            i = 0
            if not intermolecular:
                for l, v in zip(labels, values):
                    print('({}) {}\t\t{:+.5e}'.format(i, l, v))
                    i += 1

            else:
                for l, v in zip(labels, values.T):
                    print('({}) {}\t\t{:+.5e}\t{:+.5e}'.format(i, l, *v))
                    i += 1

        return labels, values

    
    def get_SOAP(self, symbols = None, intermolecular = None, connectivity = None, 
            nmax = 3, lmax = 2, rcut = 100, sigma = 1, verbose = False):
        if symbols is None:         symbols         = set(self.syst.get_chemical_symbols())
        if intermolecular is None:  intermolecular  = False

        s               = SOAP(symbols, intermolecular)
        labels, values  = s.get(self, connectivity, nmax, lmax, rcut, sigma)

        if verbose:
            print('SOAP representation:')
            print('-'*20)
            i = 0
            if not intermolecular:
                for l, v in zip(labels, values):
                    print('({}) {}\t\t{:+.5e}'.format(i, l, v))
                    i += 1

            else:
                for l, v in zip(labels, values.T):
                    print('({}) {}\t\t{:+.5e}\t{:+.5e}'.format(i, l, *v))
                    i += 1


        return labels, values

