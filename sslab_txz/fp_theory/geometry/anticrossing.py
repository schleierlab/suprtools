from typing import ClassVar, Optional

import numpy as np
import scipy.optimize
import uncertainties
from scipy.constants import c, pi
from tqdm import tqdm
from uncertainties import unumpy

from sslab_txz.fp_theory.coupling_config import CouplingConfig
from sslab_txz.fp_theory.geometry import SymmetricCavityGeometry
from sslab_txz.fp_theory.modes import ScalarModeBasis, VectorModeBasis


class AnticrossingFit:
    basis: ClassVar[VectorModeBasis] = (
        ScalarModeBasis.make_single_order_basis(0).vectorize()
        + ScalarModeBasis.make_single_order_basis(4).vectorize()
    )

    def __init__(self, modedata, ansatz_power=-1):
        """
        Parameters
        ----------
        modedata: structured numpy array
            A structured array (dtype: [q, branch, freq]) such that each
            record is an ordered triple consisting of the mode index q,
            the branch of the anticrossing feature (+1 for upper branch,
            -1 for lower branch), and freq is the mode frequency.
        ansatz_power: float, optional (default: -1)
            The frequency-dependence we assume of the coupling strength.
            The default of -1 corresponds to V ~ 1/k.
        """
        self.modedata = modedata
        self.ansatz_power = ansatz_power

    def _mixed_mode_coupling_numbers_single(
        self,
        q_branch: tuple[int, int],
        geometry: Optional[SymmetricCavityGeometry] = None,
        coupling_param: Optional[float] = None,
    ):
        """
        Returns m, delta, |g| of the coupling matrix
        [
            [m + delta  , g         ],
            [g*         , m - delta ],
        ]
        which has eigenvalues

        m +/- sqrt(delta^2 + |g|^2) = m +/- |g| sec(beta)

        where we define beta = arctan(delta/|g|)

        with corresponding eigenvectors

        [
            +/- exp(i arg(g)) sqrt((1 +/- sin(beta)) / 2),
            sqrt((1 -/+ sin(beta)) / 2),
        ]
        """
        q, _ = q_branch

        if geometry is None:
            geometry = self.geometry
        if coupling_param is None:
            coupling_param = self.upopt[4].n

        coupling_00 = geometry.coupling_matrix(
            q,
            ScalarModeBasis.make_single_order_basis(0),
            CouplingConfig.no_xcoupling,
            resonance_ratio=(1 / 2),
        )
        frequency_prediction_00 = coupling_00.eigvals.mean()
        eigvec_00x = coupling_00.eigvecs[0]

        coupling_n4 = geometry.coupling_matrix(
            q,
            ScalarModeBasis.make_single_order_basis(4),
            CouplingConfig.no_xcoupling,
            resonance_ratio=(1 / 2),
        )
        frequency_prediction_n4 = np.partition(coupling_n4.eigvals, 1)[:2].mean()
        eigvec_n4x = coupling_n4.eigvecs[0]

        frequency_halfdiff = (frequency_prediction_n4 - frequency_prediction_00) / 2
        avg_freq = (frequency_prediction_00 + frequency_prediction_n4) / 2
        avg_k = (2 * pi / c) * avg_freq

        coupling_strength = (
            coupling_param
            * geometry.fsr
            * (8 * pi * avg_k * geometry.mirror_curv_rad) ** self.ansatz_power
        )

        return np.array([avg_freq, frequency_halfdiff, coupling_strength]), eigvec_00x, eigvec_n4x

    def mixed_mode_coupling_numbers(
        self,
        q_branch,
        geometry: Optional[SymmetricCavityGeometry] = None,
        coupling_param: Optional[float] = None,
    ):
        def reduced_scalar_func(q_branch_single):
            return self._mixed_mode_coupling_numbers_single(
                q_branch_single, geometry, coupling_param
            )[0]

        coupling_numbers = np.apply_along_axis(reduced_scalar_func, -1, q_branch)
        return np.moveaxis(coupling_numbers, -1, 0)  # move last axis first

    def mixed_mode_frequency(
        self,
        q_branch,
        geometry: Optional[SymmetricCavityGeometry] = None,
        coupling_param: Optional[float] = None,
    ):
        branch = np.asarray(q_branch)[..., 1]
        avg_freq, frequency_halfdiff, coupling_strength = self.mixed_mode_coupling_numbers(
            q_branch,
            geometry,
            coupling_param,
        )
        # coupling angle:
        # return np.arctan2(frequency_halfdiff, coupling_strength)

        # eigenvalues: m +/- sqrt(g^2 + delta^2)
        return avg_freq + branch * np.sqrt(coupling_strength**2 + frequency_halfdiff**2)

    def mixed_mode_fitfunc_factory(self, pbar: Optional[tqdm] = None):
        def mixed_mode_fitfunc_vec(q_branch, length, rm, eta, p_asphere, coupling_param):
            geo = SymmetricCavityGeometry(length, rm, eta, p_asphere)

            # def reduced_scalar_func(q_branch_single):
            #     return self.mixed_mode_fitfunc_template(q_branch_single, geo, coupling_param)

            if pbar is not None:
                pbar.update(1)
                pbar.set_description(
                    f'trying L, R = {length*1e+3:.3f}, {rm*1e+3:.3f} mm; '
                    f'eta, p = {eta:.3f}, {p_asphere:.3f}, g = {coupling_param:.4f}',
                )

            # return np.apply_along_axis(reduced_scalar_func, -1, q_branch)
            return self.mixed_mode_frequency(q_branch, geo, coupling_param)

        return mixed_mode_fitfunc_vec

    def fit(self, p0=(43.7481e-3, 42.65e-3, 0.01822, 0.09, 0.1), bounds=(-np.inf, np.inf)):
        qbranch = np.array([self.modedata['q'], self.modedata['branch']]).T
        mask = ~np.isnan(self.modedata['freq'])

        with tqdm() as pbar:
            popt, pcov = scipy.optimize.curve_fit(
                self.mixed_mode_fitfunc_factory(pbar),
                qbranch[mask],
                self.modedata['freq'][mask],
                p0=p0,
                bounds=bounds,
            )
        self.upopt = uncertainties.correlated_values(popt, pcov)

    @property
    def geometry(self):
        return SymmetricCavityGeometry(*unumpy.nominal_values(self.upopt)[:4])

    def mixing_fraction(self, qbranch):
        """
        Fraction of state in the upper branch low-frequency state.

        qbranch: array_like, shape 2
            Sequence of ordered pairs (q, branch) specifying the state
            in the anticrossing feature.
        """
        qbranch = np.asarray(qbranch)

        _, frequency_halfdiff, coupling_strength = self.mixed_mode_coupling_numbers(
            qbranch,
            self.geometry,
            self.upopt[4].n,
        )

        # frequency_halfdiff is:
        # (lower-branch low-frequency freq) - (upper-branch low-frequency freq)
        mixing_angle = np.arctan2(frequency_halfdiff, coupling_strength)
        branch = qbranch[..., 1]
        return (1 - branch * np.sin(mixing_angle)) / 2

    def mixed_mode_vector(self, qbranch, relative_phase=0):
        _, eigvec_00x, eigvec_n4x = self._mixed_mode_coupling_numbers_single(qbranch)
        fraction_00 = self.mixing_fraction(qbranch)
        return np.concatenate([
            eigvec_00x * np.sqrt(fraction_00),
            eigvec_n4x * np.sqrt(1 - fraction_00) * np.exp(1j * relative_phase),
        ])
