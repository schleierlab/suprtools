'''
TODO: make most attributes regular numbers (not ufloats),
add separate support for ufloats
'''

from __future__ import annotations

import itertools
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
from numpy.typing import ArrayLike
from scipy.constants import c, epsilon_0
from scipy.constants import h as planck_h
from scipy.constants import pi
from uncertainties import unumpy as unp

import sslab_txz.fp_theory.operators as ops
from sslab_txz._typing import MaybeUFloat
from sslab_txz.fp_theory.coupling import NearConfocalCouplingMatrix
from sslab_txz.fp_theory.coupling_config import CouplingConfig
from sslab_txz.fp_theory.geometry._base import CavityGeometry
from sslab_txz.fp_theory.modes import ScalarModeBasis
from sslab_txz.fp_theory.operators import (ScalarModeOperator,
                                           VectorModeOperator)


@dataclass
class SymmetricCavityGeometry(CavityGeometry):
    length: MaybeUFloat
    mirror_curv_rad: MaybeUFloat
    eta_astig: MaybeUFloat = 0
    asphere_p: MaybeUFloat = 0  # \tilde{p}, as defined in van Exter et al. (2022), eq. 28.

    def to_nominal(self) -> SymmetricCavityGeometry:
        return SymmetricCavityGeometry(
            length=unp.nominal_values(self.length),
            mirror_curv_rad=unp.nominal_values(self.mirror_curv_rad),
            eta_astig=unp.nominal_values(self.eta_astig),
            asphere_p=unp.nominal_values(self.asphere_p),
        )

    @property
    def g(self) -> float:
        return 1 - self.length / self.mirror_curv_rad

    @property
    def z0(self) -> float:
        return unp.sqrt(self.z1 * (self.mirror_curv_rad - self.z1))

    @property
    def z1(self) -> float:
        return self.length / 2

    @property
    def rx(self) -> float:
        return self.mirror_curv_rad / (1 + self.eta_astig)

    @property
    def ry(self) -> float:
        return self.mirror_curv_rad / (1 - self.eta_astig)

    @property
    def g_x(self) -> float:
        return 1 - self.length / self.rx

    @property
    def g_y(self) -> float:
        return 1 - self.length / self.ry

    @property
    def alpha(self) -> float:
        '''
        The factor w1 / w0, where w0 is the mode waist and w1 is the
        mode spot size on the mirror. This factor is constant.
        '''
        # == w1 / w0
        return np.sqrt(self.mirror_curv_rad / (self.mirror_curv_rad - self.z1))

    def paraxial_frequency(self, longi_ind: ArrayLike, n_total: ArrayLike) -> ArrayLike:
        '''
        Parameters
        ----------
        longi_ind : array_like
            Longitudinal mode index (number of antinodes).
        n_total: array_like
            Total transverse mode index (number of transverse nodes).
            Equal to n + m for Hermite-Gauss HG(n, m) modes and to
            2p + |l| for Laguerre-Gauss LG(p, l) modes.

        Returns
        -------
        array_like
            Frequency of specified mode, in Hz, according to the
            paraxial theory
        '''
        return self.fsr * (longi_ind + (np.asarray(n_total) + 1) * unp.arccos(self.g) / pi)

    def paraxial_scalar_mode_field(self, r, z, freq):
        k = 2 * pi * freq / c
        w0 = unp.sqrt(2 * self.z0 / k)
        z_norm = z / self.z0
        w = w0 * unp.sqrt(1 + z_norm**2)

        inv_wavefront_curv = z_norm / (1 + z_norm**2) / self.z0
        gouy_phase = unp.arctan(z_norm)
        return (w0 / w) * unp.exp(-(r / w)**2) \
            * unp.cos(k*z + k * r**2 * inv_wavefront_curv / 2 - gouy_phase)

    def mode_volume(self, longi_ind: ArrayLike):
        '''
        The mode volume of a TEM(00) mode:
            L lambda z_0 / 4 = pi L w0^2 / 4

        Parameters
        ----------
        longi_ind : array_like
            The longitudinal index of the mode.

        Returns
        -------
        ndarray
            The mode volume, in m^3
        '''
        wavelength = c / self.paraxial_frequency(longi_ind, 0)
        return self.z0 * self.length * wavelength / 4

    def waist_vacuum_field(self, longi_ind: ArrayLike) -> ArrayLike:
        '''
        Electric field (rms) for a TEM(00) mode at the mode center with
        with half-photon energy in the mode. Only really makes sense for
        `longi_ind` odd.

        Parameters
        ----------
        longi_ind : array_like
            The longitudinal index of the mode.

        Returns
        -------
        ndarray
            rms vacuum field, in V/m.
        '''
        mode_volume = self.mode_volume(longi_ind)
        freq = self.paraxial_frequency(longi_ind, 0)
        return np.sqrt(planck_h * freq / (2 * epsilon_0 * mode_volume))

    @staticmethod
    def _even_modes_list(n_max):
        return list(itertools.chain(*[
            itertools.chain(*[
                [
                    (p, n_total - 2*p, s)
                    for s in ([+1, -1] if n_total != 2*p else [+1])
                ]
                for p in range(0, n_total // 2 + 1)
            ])
            for n_total in range(0, n_max+1, 2)
        ]))

    def luk_near_confocal_modes(self, q_base, max_order=6):
        even_modes = self._even_modes_list(max_order)
        paraxial_freq_00 = self.fsr * (q_base + np.arccos(self.g) / pi)
        k = 2 * pi * paraxial_freq_00 / c

        return np.array([
            self.fsr * (
                q_base
                + 0.5
                + (2*p + abs(l) + 1) * (np.arccos(self.g)/pi - 0.5)
                + 1 / (4 * pi * k * self.mirror_curv_rad) * (
                    2*p**2 + 2*p*l - l**2 + 2*p + l + 2
                    + 4*l*s - 4
                )
            )
            for (p, l, s) in even_modes
            # if not (l == 0 and s == -1)
        ])

    def coupling_matrix(
            self,
            longi_ind_base: int,
            scalar_basis: ScalarModeBasis,
            config: CouplingConfig,
            resonance_ratio: tuple[int, int] | Fraction | float | None,
    ):
        '''
        Parameters
        ----------
        resonance_ratio
            TODO
        '''
        match resonance_ratio:
            case (int(num), int(denom)):
                resonance_frac = Fraction(num, denom)
            case Fraction() as frac:
                resonance_frac = frac
            case float(x):
                resonance_frac = Fraction(x).limit_denominator(10)
            case None:
                resonance_frac = Fraction(np.arccos(self.g) / pi).limit_denominator(10)
            case _:
                raise ValueError()

        if not 0 <= resonance_frac <= 1:
            raise ValueError()

        min_basis_order = min(mode.n for mode in scalar_basis)
        min_order_mode_longi_ind = longi_ind_base \
            - resonance_frac.numerator * (min_basis_order // resonance_frac.denominator)
        paraxial_freq = self.paraxial_frequency(min_order_mode_longi_ind, min_basis_order)
        k = 2 * pi * paraxial_freq / c
        pi_k_rm = pi * k * self.mirror_curv_rad

        # *_cyc denotes operators expressed in units of cyclic phase: phase angle / (2 pi)
        h_prop_cyc = self.alpha**2 / (8 * pi_k_rm) * ops.pi_4_op
        h_wave_cyc = ((3 - self.alpha**2) * ops.rho_4 - 2 * ops.m_op) / (8 * pi_k_rm)

        h_asphere_cyc = -self.asphere_p * ops.rho_4 / (8 * pi_k_rm) \
            * self.z1 / (self.mirror_curv_rad - self.z1)

        h_vec_dimless = VectorModeOperator([
            [1 + ops.nplus - ops.nminus, 0],
            [0, 1 - ops.nplus + ops.nminus],
        ])
        h_vec_cyc = - h_vec_dimless / (2 * pi_k_rm)

        # [(operator, activation_bool, cross_coupling_bool)*]
        cyc_ops_configs: list[tuple[VectorModeOperator, bool, bool]] = [
            (h_prop_cyc.vectorize(), config.prop, config.prop_xcoupling),
            (h_wave_cyc.vectorize(), config.wave, config.wave_xcoupling),
            (h_vec_cyc, config.vec, config.vec_xcoupling),
            (h_asphere_cyc.vectorize(), config.asphere, config.asphere_xcoupling),
        ]

        h_parax_cyc: ScalarModeOperator = \
            (ops.n_op + 1) * (np.arccos(self.g)/pi - float(resonance_frac))
        matrix, vector_basis = h_parax_cyc.vectorize().toarray(scalar_basis)
        for op, include_bool, xcouple_bool in cyc_ops_configs:
            if not include_bool:
                continue

            contrib = 2 * op.toarray(scalar_basis)[0]
            if xcouple_bool:
                matrix += contrib
            else:
                matrix += np.diag(np.diagonal(contrib))

        # H_astig
        if config.astig:
            if config.astig_xcoupling:
                h_astig_dimless = ops.xi**2 - ops.eta**2
                dimless_operator = h_astig_dimless
            else:
                # this drops a LOT of off-resonant terms!
                h_astig_nocrosscouple_dimless = \
                    ops.a_plus_dag * ops.a_minus + ops.a_minus_dag * ops.a_plus
                dimless_operator = h_astig_nocrosscouple_dimless

            h_astig_cyc = self.eta_astig / (2 * pi) * np.sqrt(self.alpha**2 - 1) * dimless_operator
            matrix += 2 * h_astig_cyc.vectorize().toarray(scalar_basis)[0]

        if config.v_plus_a:
            h_v_plus_a_dimless = VectorModeOperator([
                [0, 1 - ops.nplus + ops.nminus],
                [1 + ops.nplus - ops.nminus, 0],
            ])

            h_v_plus_a = - self.eta_astig / (2 * pi_k_rm) * h_v_plus_a_dimless
            matrix += 2 * h_v_plus_a.toarray(scalar_basis)[0]

        mode_generalized_parities = np.array([
            mode.n % resonance_frac.denominator
            for mode in scalar_basis
        ])
        generalized_parity = mode_generalized_parities[0]
        if not np.all(mode_generalized_parities == generalized_parity):
            raise ValueError
        compensation_term = float(resonance_frac) * (generalized_parity + 1)

        return NearConfocalCouplingMatrix(
            cavity_geo=self,
            matrix=matrix,
            basis=vector_basis,
            longi_ind_base=longi_ind_base,
            compensation_term=compensation_term,
        )

    def near_confocal_coupling_matrix(
            self,
            longi_ind_base: int,
            config: CouplingConfig,
            max_order=6) -> NearConfocalCouplingMatrix:

        scalar_basis = ScalarModeBasis.make_transverse_mode_basis(
            range(max_order % 2, max_order + 1, 2),
        )
        return self.coupling_matrix(
            longi_ind_base,
            scalar_basis,
            config,
            resonance_ratio=(1, 2),
        )
