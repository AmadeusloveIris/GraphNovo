import re
import numpy as np

class Composition():
    __atom_mass = {
        # From NIST, "https://physics.nist.gov/cgi-bin/Compositions/stand_alone.pl?ele=&all=all&isotype=some"
        'neutron': 1.00866491595,
        'H': 1.00782503223,
        'C': 12,
        'N': 14.00307400443,
        'O': 15.99491461957,
        'P': 30.97376199842,
        'S': 31.9720711744
    }

    def __init__(self, class_input):
        if type(class_input) == str:
            formular_string = class_input
            if formular_string[0] == '-':
                self.composition = {i[0]: -int(i[1]) if i[1] else -int(1) for i in
                                    re.findall("([A-Z][a-z]?)(\d*)", formular_string)}
            else:
                self.composition = {i[0]: int(i[1]) if i[1] else int(1) for i in
                                    re.findall("([A-Z][a-z]?)(\d*)", formular_string)}
        elif type(class_input) == dict:
            self.composition = class_input
        else:
            raise TypeError
        self.mass = self.mass_calculater()

    def __add__(self, other):
        result = {}
        for k in self.composition:
            result.update({k: self.composition[k]})
        for k in other.composition:
            try:
                result[k] += other.composition[k]
                if result[k] == 0: result.pop(k)
            except KeyError:
                result.update({k: other.composition[k]})
        return Composition(result)

    def __sub__(self, other):
        result = {}
        for k in self.composition:
            result.update({k: self.composition[k]})
        for k in other.composition:
            try:
                result[k] -= other.composition[k]
                if result[k] == 0: result.pop(k)
            except KeyError:
                result.update({k: -other.composition[k]})
        return Composition(result)

    def __mul__(self, other):
        result = {}
        for k in self.composition:
            result.update({k: other * self.composition[k]})
        return Composition(result)

    def __repr__(self):
        return str(self.composition)

    def __str__(self):
        return str(self.composition)

    def mass_calculater(self):
        result = 0
        for k in self.composition:
            result += self.composition[k] * self.__atom_mass[k]
        return result

    def comp2formula(self):
        seq=''
        for k in self.composition:
            seq+=k+str(self.composition[k])
        return seq

    @classmethod
    def output_neutron(cls):
        return cls.__atom_mass['neutron']

class Residual_seq():
    __aa_residual_composition = {
        'A': Composition('C3H5ON'),
        'R': Composition('C6H12ON4'),
        'N': Composition('C4H6O2N2'),
        #'N(+.98)': Composition('C4H6O2N2') - Composition('NH3') + Composition('H2O'),
        'D': Composition('C4H5O3N'),
        #'C': Composition('C3H5ONS'),
        'c': Composition('C3H5ONS') - Composition('H') + Composition('C2H4ON'),
        'E': Composition('C5H7O3N'),
        'Q': Composition('C5H8O2N2'),
        #'Q(+.98)': Composition('C5H8O2N2') - Composition('NH3') + Composition('H2O'),
        'G': Composition('C2H3ON'),
        'H': Composition('C6H7ON3'),
        'I': Composition('C6H11ON'),
        #'L': Composition('C6H11ON'),
        'K': Composition('C6H12ON2'),
        'M': Composition('C5H9ONS'),
        'm': Composition('C5H9ONS') + Composition('O'),
        'F': Composition('C9H9ON'),
        'P': Composition('C5H7ON'),
        'S': Composition('C3H5O2N'),
        'T': Composition('C4H7O2N'),
        'W': Composition('C11H10ON2'),
        'Y': Composition('C9H9O2N'),
        'V': Composition('C5H9ON'),
    }

    def __init__(self, seqs):
        seq = [i for i in seqs]
        self.step_mass = []
        tmp = self.__aa_residual_composition[seq[0]]
        for i in seq[1:]:
            self.step_mass.append(tmp.mass_calculater())
            tmp += self.__aa_residual_composition[i]

        self.seq = seq
        self.composition = tmp
        self.mass = tmp.mass_calculater()
        self.step_mass.append(self.mass)
        self.step_mass = np.array(self.step_mass)

    def __repr__(self):
        return str(self.seq)

    def __str__(self):
        return str(self.seq)

    @classmethod
    def reset_aadict(cls,newAAdict):
        cls.__aa_residual_composition = newAAdict

    @classmethod
    def remove_from_aadict(cls, keys):
        for key in keys:
            cls.__aa_residual_composition.pop(key)

    @classmethod
    def add_to_aadict(cls, additional_AAcomps):
        for additional_AAcomp in additional_AAcomps:
            cls.__aa_residual_composition.update(additional_AAcomp)

    @classmethod
    def output_aalist(cls):
        return list(cls.__aa_residual_composition.keys())

    @classmethod
    def output_aadict(cls):
        return cls.__aa_residual_composition

    @classmethod
    def seqs2composition_list(cls,seq):
        return [cls.__aa_residual_composition[aa] for aa in seq]

class Ion():
    __ion_offset = {
        'a': Composition('-CHO'),
        'a-NH3': Composition('-CHO') + Composition('-NH3'),
        'a-H2O': Composition('-CHO') + Composition('-H2O'),
        'b': Composition('-H'),
        'b-NH3': Composition('-H') + Composition('-NH3'),
        'b-H2O': Composition('-H') + Composition('-H2O'),
        #'c': Composition('NH2'),
        #'x': Composition('CO') + Composition('-H'),
        'y': Composition('H'),
        'y-NH3': Composition('H') + Composition('-NH3'),
        'y-H2O': Composition('H') + Composition('-H2O'),
        #'z': Composition('-NH2')
    }

    __term_ion_offset = {
        'a': Composition('-CHO') + Composition('H'),
        'a-NH3': Composition('-CHO') + Composition('-NH3') + Composition('H'),
        'a-H2O': Composition('-CHO') + Composition('-H2O') + Composition('H'),
        'b': Composition('-H') + Composition('H'),
        'b-NH3': Composition('-H') + Composition('-NH3') + Composition('H'),
        'b-H2O': Composition('-H') + Composition('-H2O') + Composition('H'),
        #'c': Composition('NH2') + Composition('OH'),
        #'x': Composition('CO') + Composition('-H') + Composition('OH'),
        'y': Composition('H') + Composition('OH'),
        'y-NH3': Composition('H') + Composition('-NH3') + Composition('OH'),
        'y-H2O': Composition('H') + Composition('-H2O') + Composition('OH'),
        #'z': Composition('-NH2') + Composition('OH')
    }

    @classmethod
    def set_ionoffset_endterm(cls,nterm='H',cterm='OH'):
        result = {}
        for k in cls.__ion_offset:
            if k[0] == 'a' or k[0] == 'b' or k[0] == 'c':
                result.update({k: cls.__ion_offset[k] + Composition(nterm)})
            elif k[0] == 'x' or k[0] == 'y' or k[0] == 'z':
                result.update({k: cls.__ion_offset[k] + Composition(cterm)})
        cls.__term_ion_offset = result

    @classmethod
    def peak2sequencemz(cls, peak_mz, ion, charge=None):
        if charge==None:
            charge = int(ion[0])
            ion = ion[1:]
        return (peak_mz-Composition('H').mass_calculater())*charge-cls.__term_ion_offset[ion].mass_calculater()

    @classmethod
    def peptide2ionmz(cls, seq, ion, charge):
        ion_compsition = Residual_seq(seq).composition+cls.__term_ion_offset[ion]+Composition('H')*charge
        ion_mass = ion_compsition.mass_calculater()/charge
        return ion_mass
    
    @classmethod
    def sequencemz2ion(cls, seqmz, ion, charge=None):
        if charge==None:
            charge = int(ion[0])
            ion = ion[1:]
        return (seqmz+cls.__term_ion_offset[ion].mass_calculater())/charge+Composition('H').mass_calculater()

    @classmethod
    def precursorion2mass(cls, precursor_ion_moverz, precursor_ion_charge):
        #Composition('H2O') 是n端和c端原子的总和，但是如果做TMT或者其他对N，C端修饰的需要进行修改
        return precursor_ion_moverz*precursor_ion_charge-Composition('H2O').mass_calculater()-precursor_ion_charge*Composition('H').mass_calculater()

    @classmethod
    def add_ion(cls,ion_comps):
        for ion_comp in ion_comps:
            cls.__ion_offset.update(ion_comp)
        cls.set_ionoffset_endterm()

    @classmethod
    def remove_ion(cls, keys):
        for key in keys:
            cls.__ion_offset.pop(key)
        cls.set_ionoffset_endterm()

    @classmethod
    def reset_ions(cls, ion_comps):
        cls.__ion_offset = ion_comps
        cls.set_ionoffset_endterm()

    @classmethod
    def output_ions(cls):
        return list(cls.__ion_offset.keys())