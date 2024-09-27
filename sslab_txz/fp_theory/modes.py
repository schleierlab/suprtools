import itertools
import textwrap
from dataclasses import dataclass
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import scipy.special
from scipy.constants import pi


@dataclass(frozen=True)
class ScalarLGMode:
    nplus: int
    nminus: int

    @property
    def n(self) -> int:
        return self.nplus + self.nminus

    @property
    def p(self) -> int:
        return min(self.nplus, self.nminus)

    @property
    def ell(self) -> int:
        '''
        Azimuthal quantum number l. (Named to prevent ambiguity, per PEP 8).
        '''
        return self.nplus - self.nminus

    def __repr__(self) -> str:
        return f'|n+/- = ({self.nplus},{self.nminus})>'

    def vectorize(self, righthand):
        return VectorLGMode(self.nplus, self.nminus, righthand)

    def normed_field(self, rho, theta):
        rho = np.asarray(rho)
        theta = np.asarray(theta)

        p = self.p
        abs_l = abs(self.ell)

        p_fact = scipy.special.factorial(p)
        p_plus_l_fact = scipy.special.factorial(p + abs_l)
        prefactor = (-1)**self.p * np.sqrt(p_fact / p_plus_l_fact / pi)

        return prefactor * rho**abs_l \
            * scipy.special.assoc_laguerre(rho**2, p, k=abs_l) \
            * np.exp(-rho**2/2) * np.exp(1j * self.ell * theta)


@dataclass(frozen=True)
class VectorLGMode(ScalarLGMode):
    righthand: bool

    def __repr__(self) -> str:
        if self.righthand:
            pol_str = 'RH'
        else:
            pol_str = 'LH'

        return f'|n+/- = ({self.nplus},{self.nminus}); {pol_str}>'

    def latex(self) -> str:
        pol_str = '+' if self.righthand else '-'
        return f'|{self.nplus}{self.nminus}{pol_str}\\rangle'

    def normed_field(self, rho, theta):
        pol_sign = +1 if self.righthand else -1
        pol_vec = np.array([1, pol_sign]) / np.sqrt(2)
        return np.asarray(super().normed_field(rho, theta))[..., np.newaxis] * pol_vec


class ModeBasis(object):
    def __init__(self, modes):
        self.modes = tuple(modes)
        self.reverse_dict = {
            mode: i
            for i, mode in enumerate(self.modes)
        }

    def ind(self, mode):
        return self.reverse_dict[mode]

    def __eq__(self, other):
        return self.modes == other.modes

    def __len__(self):
        return len(self.modes)

    def __iter__(self):
        return iter(self.modes)

    def __getitem__(self, item):
        return self.modes[item]

    def __repr__(self):
        mode_reprs = (repr(mode) for mode in self.modes)
        repr_body = textwrap.indent(',\n'.join(mode_reprs), '  ')
        return f'{{\n{repr_body},\n}}'

    def eval_field(self, arr, rho, theta):
        '''
        Parameters
        ----------
        arr: (shape1, M) array_like
            array of fields to evaluate; last dimension must be same length as
            basis
        rho, theta: (shape2) array_like
            array_likes of identical shape with normalized radii and azimuthal
            angles at which to evaluate

        Returns
        -------
        (shape1, shape2) or (shape1, shape2, 1/2) array_like
            Evaluated field at supplied points for given array.
        '''

        # shape: M, shape2, (2?)
        basis_vals = np.array([f.normed_field(rho, theta) for f in self.modes])

        return np.tensordot(
            arr,  # (shape1, M)
            basis_vals,  # (M, shape2, (2?))
            axes=([-1], [0]),
        )

    def plot_field_intensity(self, vector, projection='polar', ax=None, **kwargs):
        if ax is None:
            fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))

        ax.set_aspect(1)

        if projection == 'rectilinear':
            xs = np.linspace(-6, 6, 101)
            ys = np.linspace(-6, 6, 101)
            xss, yss = np.meshgrid(xs, ys)

            rss, thetass = np.sqrt(xss**2 + yss**2), np.arctan2(yss, xss)
            avals, bvals = xss, yss
        elif projection == 'polar':
            rs = np.linspace(0, 6, 101)
            thetas = np.linspace(0, 2*pi, 200, endpoint=False)

            rss, thetass = np.meshgrid(rs, thetas)
            avals, bvals = thetass, rss
        else:
            raise ValueError

        ax.pcolormesh(
            avals,
            bvals,
            np.sum(np.abs(self.eval_field(vector, rss, thetass))**2, axis=-1),
            rasterized=True,
            **kwargs
        )


class ScalarModeBasis(ModeBasis):
    modes: Sequence[ScalarLGMode]

    @classmethod
    def make_transverse_mode_basis(cls, orders: Sequence[int]):
        modes = tuple(itertools.chain(*[
            [
                ScalarLGMode(nplus, ntot - nplus)
                for nplus in range(ntot + 1)
            ]
            for ntot in orders
        ]))

        return cls(modes)

    @classmethod
    def make_single_order_basis(cls, order):
        return cls.make_transverse_mode_basis([order])

    @classmethod
    def make_even_basis(cls, max_order):
        if max_order < 0:
            raise ValueError
        if max_order % 2 != 0:
            raise ValueError

        return cls.make_transverse_mode_basis(range(0, max_order + 1, 2))

    @classmethod
    def make_odd_basis(cls, max_order):
        if max_order < 0:
            raise ValueError
        if max_order % 2 != 1:
            raise ValueError

        return cls.make_transverse_mode_basis(range(1, max_order + 1, 2))

    def vectorize(self):
        modes_rh = (mode.vectorize(righthand=True) for mode in self.modes)
        modes_lh = (mode.vectorize(righthand=False) for mode in self.modes)
        return VectorModeBasis(itertools.chain(modes_rh, modes_lh))

    def latex(self):
        return r'|n_+ n_- \rangle'


class VectorModeBasis(ModeBasis):
    @classmethod
    def make_even_basis(cls, max_order):
        if max_order < 0:
            raise ValueError
        if max_order % 2 != 0:
            raise ValueError

        modes = tuple(itertools.chain(*[
            [
                VectorLGMode(nplus, ntot - nplus, righthand=True)
                for nplus in range(ntot + 1)
            ]
            for ntot in range(0, max_order + 1, 2)
        ])) + tuple(itertools.chain(*[
            [
                VectorLGMode(nplus, ntot - nplus, righthand=False)
                for nplus in range(ntot + 1)
            ]
            for ntot in range(0, max_order + 1, 2)
        ]))

        return cls(modes)

    def latex(self):
        return r'|n_+ n_- \pm\rangle'
