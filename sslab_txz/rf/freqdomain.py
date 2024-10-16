import copy
import datetime
import functools
from pathlib import Path
from typing import Callable, Literal, Optional, Self

import h5py
import matplotlib.gridspec
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
import scipy.signal
import skrf as rf
import uncertainties
from matplotlib.ticker import MultipleLocator
from numpy.typing import NDArray
from scipy.constants import pi
from skrf.network import Network
from tqdm import tqdm
from uncertainties import ufloat, unumpy

from sslab_txz.fp_theory.coupling import CouplingConfig
from sslab_txz.fp_theory.geometry._symmetric import SymmetricCavityGeometry
from sslab_txz.plotting import sslab_style
from sslab_txz.rf.errors import FitFailureError


class ScanDataFilter():
    def __init__(self, *, fs,  fc_hp=None, fc_lp=None, high_pass_order=0, low_pass_order=0):
        '''
        Parameters
        ----------
        fc_hp, fc_lp : scalar
            cutoff frequency for high/low-pass filter
        fs : scalar
            sampling frequency
        high_pass_order, low_pass_order: int
            high/low-pass filter order
        '''
        # nyquist_omega = pi * fs
        # high_pass_omega_c = 0.4 * nyquist_omega

        if high_pass_order > 0 and fc_hp is None:
            raise ValueError
        if low_pass_order > 0 and fc_lp is None:
            raise ValueError

        self.fc_hp = fc_hp
        self.fc_lp = fc_lp
        self.fs = fs
        self.low_pass_order = low_pass_order
        self.high_pass_order = high_pass_order

        # equivalent to this (but handles fc_hp/fc_lp being None)
        # ps = [-omega_c_hp] * high_pass_order + [-omega_c_lp] * low_pass_order
        # ks = omega_c_lp ** low_pass_order

        zs = [0] * high_pass_order
        ps = []
        ks = 1
        if high_pass_order > 0:
            omega_c_hp = 2 * pi * fc_hp
            ps += [-omega_c_hp] * high_pass_order
        if low_pass_order > 0:
            omega_c_lp = 2 * pi * fc_lp
            ps += [-omega_c_lp] * low_pass_order
            ks *= omega_c_lp ** low_pass_order

        self.zpk_s = (zs, ps, ks)
        zz, pz, kz = scipy.signal.bilinear_zpk(*self.zpk_s, fs)
        self.zpk_z = (zz, pz, kz)

        self.sos = scipy.signal.zpk2sos(zz, pz, kz)

    def filt(self, data):
        zi_step = scipy.signal.sosfilt_zi(self.sos)
        filtered, _ = scipy.signal.sosfilt(self.sos, data, zi=(data[0]*zi_step))
        return filtered

    def bode_plot(self, axs=None):
        if axs is None:
            fig, axs = plt.subplots(
                2, 1,
                sharex=True,
                constrained_layout=True,
                gridspec_kw=dict(height_ratios=[2, 1]),
            )
        ax_mag, ax_phase = axs

        nus_z, hs_z = scipy.signal.freqz_zpk(*self.zpk_z, fs=self.fs)
        omegas_s, hs_s = scipy.signal.freqs_zpk(*self.zpk_s, worN=np.geomspace(1e-7, 1e-3))

        self._make_bode_plot(*axs, omegas_s * 1e6, hs_s, label='analog')
        self._make_bode_plot(
            *axs,
            nus_z[1:] * 2 * pi * 1e6,
            hs_z[1:],
            label='digital (bilinear xform)',
        )

        ax_mag.set_ylabel('Magnitude (dB)')
        ax_phase.set_ylabel('Phase (deg)')
        ax_phase.set_xlabel('"Frequency" (1/MHz)')

        for ax in axs:
            sslab_style(ax)
            if self.fc_hp is not None:
                ax.axvline(
                    1e6 * self.fc_hp,
                    linestyle='dashed',
                    color='0.5',
                    label=f'$f_{{c,\\mathrm{{HP}}}}^{{-1}} = {1e-6/self.fc_hp:.3f}$ MHz',
                )
            if self.fc_lp is not None:
                ax.axvline(
                    1e6 * self.fc_lp,
                    linestyle='dotted',
                    color='0.5',
                    label=f'$f_{{c,\\mathrm{{LP}}}}^{{-1}} = {1e-6/self.fc_lp:.3f}$ MHz',
                )
            ax.axvline(
                1e6 * self.fs / 2,
                linestyle='dashdot',
                color='0.5',
                label=f'Nyquist frequency 1/({2/self.fs / 1e3:.1f} kHz)',
            )

        ax_mag.legend()
        ax_phase.yaxis.set_major_locator(MultipleLocator(30))
        ax_phase.yaxis.set_minor_locator(MultipleLocator(15))

        secax = ax_mag.secondary_xaxis('top', functions=(lambda x: 1/x, lambda x: 1/x))
        secax.set_xlabel('"Wavelength" (MHz)')

    def latex_repr(self):
        inv = '^{-1}'
        if self.high_pass_order == 0:
            fclp = R'f_{c, \mathrm{LP}}'
            return f'${fclp}{inv} = {1e-6/self.fc_lp:.3f}$ MHz'
        if self.low_pass_order == 0:
            fchp = R'f_{c, \mathrm{HP}}'
            return f'${fchp}{inv} = {1e-6/self.fc_hp:.3f}$ MHz'

        fchlp = R'f_{c, \mathrm{HP,LP}}'
        return f'${fchlp}{inv} = {1e-6/self.fc_hp:.3f}, {1e-6/self.fc_hp:.3f}$ MHz'

    @staticmethod
    def _make_bode_plot(ax_mag, ax_phase, omegas, hs, **kwargs):
        ax_mag.semilogx(
            omegas / (2 * pi),
            rf.complex_2_db(hs),
            **kwargs,
        )
        ax_phase.semilogx(
            omegas / (2 * pi),
            np.unwrap(rf.complex_2_degree(hs), period=360),
            **kwargs,
        )


class WideScanNetwork(rf.Network):
    def __init__(self, *args, **kwargs):
        if len(args) == 1 and isinstance(args[0], rf.Network):
            network = args[0]
            network.frequency
            super().__init__(
                frequency=network.frequency,
                s=network.s,
                name=network.name,
            )
        else:
            super().__init__(*args, **kwargs)

    @property
    def freq_step(self):
        assert self.frequency.sweep_type == 'lin'
        return self.frequency.df[0]

    def make_filter(self, lambda_hp=None, lambda_lp=None, high_pass_order=0, low_pass_order=0):
        pna_sampling_freq = 1 / self.freq_step
        return ScanDataFilter(
            fs=pna_sampling_freq,
            fc_hp=(None if lambda_hp is None else 1/lambda_hp),
            fc_lp=(None if lambda_lp is None else 1/lambda_lp),
            high_pass_order=high_pass_order,
            low_pass_order=low_pass_order,
        )

    def _subnetwork(self, center_ghz, span_ghz):
        halfspan_ghz = span_ghz / 2

        left_endpt = round(center_ghz - halfspan_ghz, ndigits=10)
        right_endpt = round(center_ghz + halfspan_ghz, ndigits=10)
        index_str = f'{left_endpt}-{right_endpt}ghz'

        return self[index_str]

    def plot_filtered_network(
        self,
        filt: ScanDataFilter,
        fig=None,
        axs=None,
        **kwargs,
    ):
        make_fig = fig is None or axs is None
        if make_fig:
            fig, axs = plt.subplots(figsize=(9, 3.6), constrained_layout=True, nrows=2, sharex=True)
        for ax in axs:
            sslab_style(ax)

        filtered_network = filt.filt(self.s.flatten())

        ax1, ax2 = axs
        ax1.plot(
            self.f / 1e+9,
            rf.complex_2_magnitude(self.s.flatten()),
            linewidth=0.75,
            label=self.name,
            **kwargs,
        )
        ax2.plot(
            self.f / 1e+9,
            rf.complex_2_magnitude(filtered_network),
            linewidth=0.25,
            label=self.name,
            **kwargs,
        )

        if make_fig:
            fig.supxlabel('Frequency (GHz)')
            ax1.set_ylabel('Raw data')
            ax2.set_ylabel(fr'Filtered, {filt.latex_repr()}')

        # fig.suptitle(fr'$T_{{\mathrm{{plat, samp}}}} =
        # {base_t_mat["T_plat"][0,0], base_t_mat["T_sample"][0,0]}$ K')

        return fig, axs

    def fsr_compare_plots(
            self,
            center_freq_ghz,
            span_ghz,
            fsr_guess_ghz: float,
            offset_range,
            scale=1,
            geo: Optional[SymmetricCavityGeometry] = None,
            filt: Optional[ScanDataFilter] = None,
            fig=None,
            axs=None,
            **kwargs):
        min_offset, max_offset = offset_range
        nrows_base = max_offset - min_offset + 1

        if filt is not None:
            row_factor = 2
            hratios = [2, 1] * nrows_base
        else:
            row_factor = 1
            hratios = [1] * nrows_base

        nrows = nrows_base * row_factor

        if fig is None and axs is None:
            fig, axs = plt.subplots(
                figsize=(12, 1.2*nrows),
                nrows=nrows,
                gridspec_kw=dict(height_ratios=hratios),
                sharex='col',
                constrained_layout=True,
            )
            fig.supxlabel('Frequency [GHz]')
            for ax in axs:
                sslab_style(ax)
        elif (fig is not None) and (axs is not None):
            pass
        else:
            raise ValueError

        unfiltered_axs = axs
        if filt is not None:
            unfiltered_axs = axs[0::2]

        offset_iter = range(min_offset, max_offset + 1)

        for ax, offset in zip(unfiltered_axs, offset_iter):
            offset_center_freq_ghz = center_freq_ghz + offset * fsr_guess_ghz

            subnet = self._subnetwork(offset_center_freq_ghz, span_ghz)
            ax.plot(
                subnet.f / 1e9 - fsr_guess_ghz * offset,
                rf.complex_2_db(subnet.s.flatten() * scale),
                label=fr'${offset:+d}\times {fsr_guess_ghz}$ GHz',
                rasterized=True,
                **kwargs,
            )

        if filt is not None:
            filtered_axs = axs[1::2]
            for ax, offset in zip(filtered_axs, offset_iter):
                offset_center_freq_ghz = center_freq_ghz + offset * fsr_guess_ghz
                subnet = self._subnetwork(offset_center_freq_ghz, span_ghz)

                filtered_magnitude = \
                    rf.complex_2_magnitude(filt.filt(subnet.s[:, 0, 0] * scale))
                # noise_floor = np.quantile(filtered_magnitude, 0.99)
                # peak_inds, _ = scipy.signal.find_peaks(
                #     filtered_magnitude,
                #     prominence=2*noise_floor,
                #     distance=int(2e6//self.freq_step),
                # )
                ax.plot(
                    subnet.f / 1e9 - fsr_guess_ghz * offset,
                    filtered_magnitude,
                    label=fr'${offset:+d}\times {fsr_guess_ghz}$ GHz',
                    rasterized=True,
                    **kwargs,
                )
                # ax.plot(
                #     subnet.f[peak_inds] / 1e9 - fsr_guess_ghz * offset,
                #     filtered_magnitude[peak_inds],
                #     marker='x',
                #     color='C1',
                #     linestyle='None',
                # )

            for ax_db, ax_filt, offset in zip(unfiltered_axs, filtered_axs, offset_iter):
                ax_db.set_ylabel(f'$|S_{{21}}|$ (dB)\n[+${offset}\\times$ FSR]')
                ax_filt.set_ylabel("filtered\n(linear)")

        if geo is not None:
            for ax_group, offset_ind in zip(axs.reshape(len(offset_iter), -1), offset_iter):
                q_base = int(center_freq_ghz // fsr_guess_ghz) + offset_ind
                inds = slice(None, None, 2)
                freq_offset = offset_ind * fsr_guess_ghz * 1e+9

                diag_result = geo.near_confocal_coupling_matrix(
                    q_base,
                    CouplingConfig.no_xcoupling,
                    max_order=8,
                )

                for i, ax in enumerate(ax_group):
                    diag_result.annotate_modes(
                        inds,
                        offset=freq_offset,
                        scaling=1e9,
                        ax=ax,
                        label=(0.5 if i == len(ax_group) - 1 else False),  # label the last Axes
                        # color=mode_colorfunc,
                        # **annotate_kwargs,
                    )
                inset_kw_default = dict(
                    gap=24e+6,
                    inset_size=0.3,
                    inset_pad=0.18,
                    rasterized=True,
                )
                inset_kwargs = {**inset_kw_default}  # , **inset_kw}
                diag_result.plot_mode_insets(
                    inds,
                    offset=freq_offset,
                    scaling=1e9,
                    fig=fig,
                    ax=ax_group[0],
                    projection='polar',
                    **inset_kwargs,
                )

        if filt is None:
            fig.supylabel('$S_{21}$ [dB]')

        return fig, axs

    def fit_network(self, n_poles_cmplx: Optional[int] = None):
        return fit_network(self, n_poles_cmplx)

    def fit_narrow_mode(
            self,
            n_poles_cmplx: Optional[int] = None,
            frequency_err_max: float = 400e+3,
    ):
        return fit_narrow_mode(self, n_poles_cmplx, frequency_err_max)


class WideScanData():
    s11: Optional[WideScanNetwork]
    s21: Optional[WideScanNetwork]

    def __init__(self, s11, s21, metadata):
        self.s11 = s11
        self.s21 = s21
        self.metadata = metadata

    @staticmethod
    def _get_s_param(
            f: h5py.File,
            param: Literal['s11', 's21'],
            raw: bool = False) -> Optional[NDArray[np.complex128]]:

        if raw:
            datapath = f'data/s_params_raw/{param}'
        else:
            datapath = f'data/s_params/{param}'

        if datapath not in f:
            return None

        real_part = np.array(f.get(f'{datapath}/real'))
        imag_part = np.array(f.get(f'{datapath}/imag'))
        return real_part + 1j*imag_part

    @classmethod
    def _load_window_data(
            cls,
            f: h5py.File,
            sample_temps: bool = False,
            raw: bool = False,
    ) -> tuple[
            rf.Frequency,
            Optional[NDArray[np.complex128]],
            Optional[NDArray[np.complex128]],
            tuple,
    ]:
        frequencies_dataset = f.get('data/frequencies')
        freq_obj = rf.Frequency.from_f(frequencies_dataset, unit='Hz')
        s11_arr = cls._get_s_param(f, 's11', raw=raw)
        s21_arr = cls._get_s_param(f, 's21', raw=raw)

        end_time_str = f.attrs['run_time']
        end_datetime = datetime.datetime.strptime(end_time_str, '%Y%m%dT%H%M%S') \
            .replace(tzinfo=datetime.timezone.utc)

        metadata_pt: tuple[float, float, datetime.datetime] | tuple[float, datetime.datetime]

        temperature_attr = (
            'temperature_still' if 'temperature_still' in f.attrs
            else 'temperature_plate'
        )

        if sample_temps:
            metadata_pt = (
                f.attrs[temperature_attr],
                f.attrs['temperature_sample'],
                end_datetime,
            )
        else:
            metadata_pt = (
                f.attrs[temperature_attr],
                end_datetime,
            )

        return freq_obj, s11_arr, s21_arr, metadata_pt

    @classmethod
    def from_window(
        cls,
        h5_path: Path,
        network_name: str,
        sample_temps: bool = False,
    ):
        with h5py.File(h5_path) as f:
            freq_obj, s11_arr, s21_arr, metadata_pt = \
                cls._load_window_data(f, sample_temps=sample_temps)

        s11_net = None
        if s11_arr is not None:
            s11_net = WideScanNetwork(
                frequency=freq_obj,
                s=s11_arr,
                name=f'{network_name}, S11',
            )

        s21_net = None
        if s21_arr is not None:
            s21_net = WideScanNetwork(
                frequency=freq_obj,
                s=s21_arr,
                name=f'{network_name}, S21',
            )

        names: tuple[str, ...]
        if sample_temps:
            names = ('t_plate', 't_samp', 'end_time')
        else:
            names = ('t_plate', 'end_time')

        # formats=None is a workaround for https://github.com/numpy/numpy/issues/26376
        metadata_arr = np.rec.fromrecords((metadata_pt,), names=names, formats=None)

        return cls(s11=s11_net, s21=s21_net, metadata=metadata_arr)

    @classmethod
    def from_windows_nonconsec(
        cls,
        h5s_path: Path,
        network_name: str,
        sample_temps: bool = False,
    ):
        window_files = h5s_path.glob('window*.h5')
        n = max(int(p.name[6:-3]) for p in window_files) + 1

        wide_scan_datas = []
        for i in tqdm(range(n)):
            h5_path = h5s_path / f'window{i:03d}.h5'
            metadata_record = cls.from_window(h5_path, network_name, sample_temps=sample_temps)
            wide_scan_datas.append(metadata_record)

        metadata = [d.metadata for d in wide_scan_datas]
        stitched_s11 = WideScanNetwork(
            functools.reduce(
                rf.network.stitch,
                (d.s11 for d in wide_scan_datas),
            ),
        )
        stitched_s21 = WideScanNetwork(
            functools.reduce(
                rf.network.stitch,
                (d.s21 for d in wide_scan_datas),
            ),
        )
        stitched_s11.drop_non_monotonic_increasing()
        stitched_s21.drop_non_monotonic_increasing()

        return cls(s11=stitched_s11, s21=stitched_s21, metadata=metadata)

    @classmethod
    def from_windows(
        cls,
        h5s_path: Path,
        network_name: str,
        sample_temps: bool = False,
        raw: bool = False,
    ):
        window_files = h5s_path.glob('window*.h5')
        n = max(int(p.name[6:-3]) for p in window_files) + 1

        with h5py.File(h5s_path / 'window000.h5') as f:
            frequencies_dataset = f.get('data/frequencies')
            start_freq = frequencies_dataset[0]
            points_per_window = len(frequencies_dataset)
        with h5py.File(h5s_path / f'window{n-1:03d}.h5') as f:
            stop_freq = f.get('data/frequencies')[-1]

        n_points = n * (points_per_window - 1) + 1

        s11_arr = np.empty(n_points, np.csingle)
        s21_arr = np.empty(n_points, np.csingle)

        metadata = []

        for i in tqdm(range(n)):
            start_ind = i * (points_per_window - 1)
            end_ind = start_ind + points_per_window
            with h5py.File(h5s_path / f'window{i:03d}.h5') as f:
                _, s11_data, s21_data, metadata_pt = \
                    cls._load_window_data(f, sample_temps=sample_temps, raw=raw)
                s11_arr[start_ind:end_ind] = s11_data
                s21_arr[start_ind:end_ind] = s21_data
                metadata.append(metadata_pt)

        freq_obj = rf.Frequency(start_freq / 1e9, stop_freq / 1e9, n_points, 'GHz')

        s11_net = WideScanNetwork(
            frequency=freq_obj,
            s=s11_arr,
            name=f'{network_name}, S11',
        )
        s21_net = WideScanNetwork(
            frequency=freq_obj,
            s=s21_arr,
            name=f'{network_name}, S21',
        )

        if sample_temps:
            names = ['t_plate', 't_samp', 'end_time']
        else:
            names = ['t_plate', 'end_time']

        # formats=None is a workaround for https://github.com/numpy/numpy/issues/26376
        metadata_arr = np.rec.fromrecords(metadata, names=names, formats=None)
        return cls(s11=s11_net, s21=s21_net, metadata=metadata_arr)

    def __add__(self, other):
        s11 = WideScanNetwork(rf.stitch(self.s11, other.s11))
        s21 = WideScanNetwork(rf.stitch(self.s21, other.s21))
        s11.drop_non_monotonic_increasing()
        s21.drop_non_monotonic_increasing()

        metadata = np.concatenate((self.metadata, other.metadata))
        return WideScanData(s11, s21, metadata)


RealImagDict = dict[Literal['real', 'imag'], NDArray]
RefinedParamsDict = dict[Literal['poles', 'residues', 'constants'], RealImagDict]


class VectorFittingFancy(rf.VectorFitting):
    refined_fit_params: Optional[RefinedParamsDict] = None
    refined_model: Optional[Self] = None

    def __init__(self, network: Network):
        super().__init__(network)

    @staticmethod
    def _setup_iq_ax(ax, scale_str):
        ax.set_aspect(1)
        ax.set_xlabel(fr'$\mathrm{{Re}}(S_{{21}}) \times {scale_str}$')
        ax.set_ylabel(fr'$\mathrm{{Im}}(S_{{21}}) \times {scale_str}$')

    @staticmethod
    def _get_offset_and_func(x, unit):
        '''
        '''
        rounding = 1e+3 * unit
        offset = (x // rounding) * rounding

        offset_unit = 10 ** (3 * (np.log10(x) // 3))
        offset_scaled = offset / offset_unit

        def offset_func(f):
            return (f - offset) / unit

        return offset_scaled, offset_unit, offset_func

    @property
    def fwhms(self):
        pole_real_parts = np.real(self.poles)
        return -2 * pole_real_parts / (2 * pi)

    @property
    def refined_fwhms(self):
        assert self.refined_fit_params is not None
        pole_real_parts = self.refined_fit_params['poles']['real']
        return -2 * pole_real_parts / (2 * pi)

    @property
    def resonances(self):
        pole_imag_parts = np.imag(self.poles)
        return pole_imag_parts / (2 * pi)

    @property
    def refined_resonances(self):
        assert self.refined_fit_params is not None
        pole_imag_parts = self.refined_fit_params['poles']['imag']
        return pole_imag_parts / (2 * pi)

    @property
    def refined_residue_mags(self):
        if self.refined_fit_params is None:
            raise RuntimeError

        residue_mag_angular = unumpy.sqrt(
            self.refined_fit_params['residues']['real']**2
            + self.refined_fit_params['residues']['imag']**2
        )
        return residue_mag_angular / (2 * pi)

    def closest_pole_uparams(self, frequency: float, frequency_err_max: float = 400e+3):
        '''
        Parameters of the low-frequency-uncertainty partial fraction
        closest in frequency to a specified value. A refined fit must
        already exist.

        Parameters
        ----------
        frequency: float
            Nominal frequency to center search around
        frequency_err_max: float, optional
            Maximum frequency error for a partial fraction contribution
            to be considered.

        Returns
        -------
        tuple[ufloat, ufloat, ufloat, ufloat]
            Re(p), Im(p), Re(r), Im(r), where p and r are respectively
            the pole and residue corresponding to the partial fraction
            in the fit closest to the specified frequency.
        '''
        if self.refined_fit_params is None:
            raise ValueError

        localized_mode_mask = (unumpy.std_devs(self.refined_resonances) < frequency_err_max)
        nominal_freqs = unumpy.nominal_values(self.refined_resonances)
        masked_distances = np.where(
            localized_mode_mask,
            np.abs(nominal_freqs - frequency),
            np.inf,
        )
        mode_ind = np.argmin(masked_distances)

        return (
            self.refined_fit_params['poles']['real'][mode_ind],
            self.refined_fit_params['poles']['imag'][mode_ind],
            self.refined_fit_params['residues']['real'][mode_ind],
            self.refined_fit_params['residues']['imag'][mode_ind],
        )

    def print_poles(self, freq_prec=3, fwhm_prec=4):
        print('Freq (GHz)\tFWHM (MHz)')
        for pole in sorted(self.poles, key=np.imag):
            pole_freq_ghz = np.imag(pole) / (2 * pi * 1e9)
            pole_freq_fmt_str = f'{{:5.{freq_prec}f}}'
            pole_freq_str = pole_freq_fmt_str.format(pole_freq_ghz)
            pole_fwhm_fmt_str = f'{{:9.{fwhm_prec}f}}'
            pole_fwhm_str = pole_fwhm_fmt_str.format(-2 * np.real(pole) / (2 * pi * 1e6))
            print(f'{pole_freq_str}\t\t{pole_fwhm_str}')

    @staticmethod
    def _format_ufreq(ufreq):
        thousands = np.log10(ufreq.n) // 3
        if thousands == 4:
            unit = 'THz'
        elif thousands == 3:
            unit = 'GHz'
        elif thousands == 2:
            unit = 'MHz'
        elif thousands == 1:
            unit = 'kHz'
        elif thousands <= 0:
            unit = 'Hz'
            thousands = 0

        fwhm_significand = ufreq / (1e+3)**thousands
        return f'{fwhm_significand:S} {unit}'

    def print_refined_poles(self):
        print(f'{"Freq (GHz)":16}{"FWHM":^24}{"Residue mag":^24}')
        for i in range(len(self.refined_resonances)):
            freq = self.refined_resonances[i]
            fwhm = self.refined_fwhms[i]
            resmag = self.refined_residue_mags[i]

            fwhm_str = self._format_ufreq(fwhm)
            resmag_str = self._format_ufreq(resmag)
            print(f'{freq/1e+9:16S}{fwhm_str:>24}{resmag_str:>24}')

    def visualize(self, plot_unit=None, polar_scale=1e3):
        fig = plt.figure(
            figsize=(9.6, 3.6),
            constrained_layout=True,
        )
        gs = matplotlib.gridspec.GridSpec(
            nrows=2,
            ncols=3,
            figure=fig,
            height_ratios=[2, 1],
        )

        ax_mag = fig.add_subplot(gs[0, 0])
        ax_mag.tick_params('x', labelbottom=False)
        ax_phase = fig.add_subplot(gs[1, 0], sharex=ax_mag)
        ax_iq = fig.add_subplot(gs[:, 2], projection='polar')
        ax_iq_zoom = fig.add_subplot(gs[:, 1])

        for ax in ax_mag, ax_phase, ax_iq_zoom:
            sslab_style(ax)

        ax_mag.set_ylabel('$S_{21}$ [dB]')
        ax_phase.set_ylabel('Phase [deg]')

        polar_scale_log = np.log10(polar_scale)
        if polar_scale_log % 1 == 0:
            polar_scale_latex_str = f'10^{{{int(polar_scale_log)}}}'
        else:
            raise NotImplementedError('only powers of 10 supported')

        self._setup_iq_ax(ax_iq_zoom, polar_scale_latex_str)

        freq_obj = self.network.frequency

        if plot_unit is None:
            plot_unit = self._default_plot_unit()

        xoffset, offset_multiplier, _ = self._get_offset_and_func(
            freq_obj.center,
            freq_obj.multiplier_dict[plot_unit],
        )

        if offset_multiplier == 1:
            offset_unit_str = 'Hz'
        elif offset_multiplier == 1e+3:
            offset_unit_str = 'kHz'
        elif offset_multiplier == 1e+6:
            offset_unit_str = 'MHz'
        elif offset_multiplier == 1e+9:
            offset_unit_str = 'GHz'

        plot_unit_str = freq_obj.unit_dict[plot_unit]
        freq_label = f'$\\mathrm{{Frequency}} - {xoffset}$ {offset_unit_str} [{plot_unit_str}]'
        ax_phase.set_xlabel(freq_label)

        def major_formatter(x, pos):
            three_digit_str = f'{x % 1e3:g}'
            if x < 0:
                return f'[{three_digit_str}]'
            return three_digit_str
        ax_phase.xaxis.set_major_formatter(major_formatter)

        data_color = 'C0'
        model_color = 'red'

        marker_plot_kwargs = dict(
            marker='.',
            markersize=1,
            linestyle='None',
            color=data_color,
        )

        model_plot_kwargs = dict(
            linewidth=1.5,
            alpha=0.5,
            color=model_color,
            label=f'Vector fit (RMS error {self.get_rms_error(0, 0):.3E})',
        )

        self.plot_model_db(
            ax=ax_mag,
            model_plot_kw=model_plot_kwargs,
            data_plot_kw=marker_plot_kwargs,
        )
        self.plot_model_deg_unwrap(
            ax=ax_phase,
            model_plot_kw=model_plot_kwargs,
            data_plot_kw=marker_plot_kwargs,
        )

        iq_model_plot_kw = dict(
            marker='x',
            linestyle='None',
            markersize=1,
            color=model_color,
        )

        iq_data_plot_kw = dict(
            marker='.',
            markersize=1,
            alpha=0.3,
            # linestyle='None',
            linewidth=1,
            color=data_color,
        )

        self.plot_model_polar(
            ax=ax_iq,
            model_plot_kw=iq_model_plot_kw,
            data_plot_kw=iq_data_plot_kw,
            scale=polar_scale,
        )
        self.plot_model_cartesian(
            ax=ax_iq_zoom,
            model_plot_kw=iq_model_plot_kw,
            data_plot_kw=iq_data_plot_kw,
            scale=polar_scale,
        )

        ax_iq.set_title(fr'$S_{{21}} \times {polar_scale_latex_str}$')

        fig.ax_mag = ax_mag
        fig.ax_phase = ax_phase
        fig.ax_iq = ax_iq
        fig.ax_iq_zoom = ax_iq_zoom
        return fig

    def _default_plot_unit(self):
        freq_obj = self.network.frequency
        if freq_obj.span > 1e+6:
            return 'mhz'
        elif freq_obj.span > 1e+3:
            return 'khz'
        elif freq_obj.span > 1e+0:
            return 'hz'

    def plot_model_scalar(
            self,
            converter,
            ax=None,
            data_plot_kw=dict(),
            model_plot_kw=dict(),
            refined_plot_kw=None,
            plot_unit=None):
        if ax is None:
            _, ax = plt.subplots()

        freq_obj = self.network.frequency
        if plot_unit is None:
            plot_unit = self._default_plot_unit()

        _, _, freq_to_xunit = self._get_offset_and_func(
            freq_obj.center,
            freq_obj.multiplier_dict[plot_unit],
        )

        freq_min, freq_max = self.network.frequency.f[[0, -1]]
        freq_range = np.linspace(freq_min, freq_max, 5001)
        offset_freq_range = freq_to_xunit(freq_range)

        ax.plot(
            freq_to_xunit(self.network.frequency.f),
            converter(self.network.s.flatten()),
            **data_plot_kw,
        )

        if self.refined_model is None:
            model_response = self.get_model_response(0, 0, freqs=freq_range)
            ax.plot(
                offset_freq_range,
                converter(model_response),
                **model_plot_kw,
            )
        else:
            refined_response = self.refined_model.get_model_response(0, 0, freqs=freq_range)

            if refined_plot_kw is None:
                refined_plot_kw = copy.copy(model_plot_kw)
                refined_plot_kw['color'] = 'red'
                refined_plot_kw['alpha'] = 0.5
                refined_plot_kw['label'] = \
                    f'Refined (RMS error {self.refined_model.get_rms_error(0, 0):.3E})'
            ax.plot(
                offset_freq_range,
                converter(refined_response),
                **refined_plot_kw,
            )

    def plot_model_db(self, ax=None, data_plot_kw=dict(), model_plot_kw=dict(), plot_unit=None):
        if ax is None:
            _, ax = plt.subplots()

        self.plot_model_scalar(
            rf.mathFunctions.complex_2_db,
            ax=ax,
            model_plot_kw=model_plot_kw,
            data_plot_kw=data_plot_kw,
        )

    def plot_model_deg_unwrap(
            self,
            ax=None,
            data_plot_kw=dict(),
            model_plot_kw=dict(),
            plot_unit=None):
        if ax is None:
            _, ax = plt.subplots()

        self.plot_model_scalar(
            lambda s: np.unwrap(rf.mathFunctions.complex_2_degree(s), period=360),
            ax=ax,
            data_plot_kw=data_plot_kw,
            model_plot_kw=model_plot_kw,
        )

    def plot_model_parametric(
            self,
            x_converter,
            y_converter,
            ax,
            data_plot_kw=dict(),
            model_plot_kw=dict(),
            refined_plot_kw=None,
            scale=1):

        scaled_model_response = scale \
            * self.get_model_response(0, 0, freqs=self.network.frequency.f)
        scaled_sparam = self.network.s.flatten() * scale

        ax.plot(
            x_converter(scaled_sparam),
            y_converter(scaled_sparam),
            **data_plot_kw,
        )
        if self.refined_model is None:
            ax.plot(
                x_converter(scaled_model_response),
                y_converter(scaled_model_response),
                **model_plot_kw,
            )
        else:
            refined_response_scaled = scale \
                * self.refined_model.get_model_response(0, 0, freqs=self.network.frequency.f)

            if refined_plot_kw is None:
                refined_plot_kw = copy.copy(model_plot_kw)
                refined_plot_kw['color'] = 'red'
                refined_plot_kw['alpha'] = 0.7
            ax.plot(
                x_converter(refined_response_scaled),
                y_converter(refined_response_scaled),
                **refined_plot_kw,
            )

    def plot_model_polar(self, ax=None, data_plot_kw=dict(), model_plot_kw=dict(), scale=1):
        if ax is None:
            _, ax = plt.subplots(subplot_kw=dict(projection='polar'))

        self.plot_model_parametric(
            rf.complex_2_radian,
            rf.complex_2_magnitude,
            ax,
            data_plot_kw=data_plot_kw,
            model_plot_kw=model_plot_kw,
            scale=scale,
        )

    def plot_model_cartesian(self, ax=None, model_plot_kw=dict(), data_plot_kw=dict(), scale=1):
        if ax is None:
            _, ax = plt.subplots()

        self.plot_model_parametric(
            np.real,
            np.imag,
            ax,
            data_plot_kw=data_plot_kw,
            model_plot_kw=model_plot_kw,
            scale=scale,
        )

        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()

        xcent, ycent = (xmin + xmax) / 2, (ymin + ymax)/2
        maxspan = max(xmax - xmin, ymax - ymin)
        halfmaxspan = maxspan / 2

        ax.set_xlim(xcent - halfmaxspan, xcent + halfmaxspan)
        ax.set_ylim(ycent - halfmaxspan, ycent + halfmaxspan)

    def _get_norm(self, norm: bool | float) -> float:
        match norm:
            case bool(b):
                return np.mean(self.network.f) if b else 1
            case float(n):
                return n
            case _:
                raise ValueError()

    def unravel_fit_params(self, norm: bool | float = True):
        '''
        Unravel the fit parameters in the following canonical ordering:

        real poles
        complex poles, real part
        complex poles, imaginary part
        residues of real poles, real part
        residues of complex poles, real part
        residues of real poles, imaginary part
        residues of complex poles, imaginary part
        constant coefficient (real), if nonzero
        proportional coefficient (real), if nonzero

        Parameters
        ----------
        norm : float or bool, optional
            Controls whether we return poles and residues with some normalization.
            If a float, gives the normalization in rad/s. Then

            returned_poles = actual_poles / norm
            returned_residues = actual_residues / norm
            returned_propterm = actual_propterm * norm

            If True, normalize to the mean network frequency. If False, do not normalize.

        Returns
        -------
        ndarray
            unraveled (real) fit parameters
        '''
        # TODO extend beyond 1-port networks
        if self.network.nports > 1:
            raise ValueError('Only 1-port networks supported.')

        norm_val = self._get_norm(norm)
        poles = self.poles / norm_val
        residues = self.residues[0] / norm_val
        constants = self.constant_coeff
        props = self.proportional_coeff * norm_val

        real_mask = (np.imag(poles) == 0)
        return np.hstack((
            np.real(poles[real_mask]),
            np.real(poles[~real_mask]),
            np.imag(poles[~real_mask]),
            np.real(residues[real_mask]),
            np.real(residues[~real_mask]),
            np.imag(residues[~real_mask]),
            constants[constants != 0],
            props[props != 0],
        ))

    def refine_fit(
            self,
            sigma=None,
            absolute_sigma=False,
            norm=True):
        fit_function_input = np.array(np.meshgrid(self.network.f, [0, 1])).reshape(2, -1).T
        fit_function_values = np.hstack((
            np.real(self.network.s.flatten()),
            np.imag(self.network.s.flatten()),
        ))

        p0 = self.unravel_fit_params(norm=norm)
        bounds = np.sort((0.5*p0, 1.5*p0), axis=0)
        n_poles_real = np.sum(np.imag(self.poles) == 0)
        n_poles_cmplx = len(self.poles) - n_poles_real
        constant_coeff = bool(self.constant_coeff[0])
        proportional_coeff = bool(self.proportional_coeff[0])

        fit_sigma = None
        if sigma is not None:
            sigma = np.asarray(sigma)
            match sigma.ndim:
                case 0:
                    fit_sigma = sigma * np.ones_like(fit_function_values)
                case 1 | 2:
                    fit_sigma = sigma
                case _:
                    raise ValueError()

        fit_function, jacobian = self.make_fit_function(
            n_poles_real,
            n_poles_cmplx,
            constant_coeff,
            proportional_coeff,
            norm=norm,
        )

        try:
            popt, pcov, *extra_info = scipy.optimize.curve_fit(
                fit_function,
                fit_function_input,
                fit_function_values,
                sigma=fit_sigma,
                absolute_sigma=absolute_sigma,
                p0=p0,
                bounds=bounds,
                xtol=1e-12,
                jac=jacobian,
                full_output=True,
            )
        except RuntimeError:
            raise FitFailureError

        raveler = self.make_fit_param_raveler(
            n_poles_real,
            n_poles_cmplx,
            constant_coeff,
            proportional_coeff,
            break_off_real_poles=False,
            complex_values=False,
            norm=norm,
        )

        refined_params = raveler(uncertainties.correlated_values(popt, pcov))
        self.refined_fit_params = refined_params

        new_vf = VectorFittingFancy(self.network)
        new_vf.poles = unumpy.nominal_values(refined_params['poles']['real']) \
            + unumpy.nominal_values(refined_params['poles']['imag']) * 1j
        new_vf.residues = (
            unumpy.nominal_values(refined_params['residues']['real'])
            + unumpy.nominal_values(refined_params['residues']['imag']) * 1j)[np.newaxis, :]
        new_vf.constant_coeff = unumpy.nominal_values([refined_params.get('constants', 0)])
        new_vf.proportional_coeff = unumpy.nominal_values([refined_params.get('proportional', 0)])

        self.refined_model = new_vf
        return new_vf

    def make_fit_param_raveler(
            self,
            n_poles_real: int,
            n_poles_cmplx: int,
            constant_coeff: bool,
            proportional_coeff: bool,
            *,
            norm: bool | float = True,
            break_off_real_poles: bool,
            complex_values: bool) -> Callable:
        '''
        Create a function that takes in the canonical Vector Fit
        parameter ordering (defined in `unravel_fit_params`) and returns
        them grouped in a dict.

        Parameters
        ----------
        n_poles_real, n_poles_cmplx: int
            Number of real and complex poles, respectively.
        constant_coeff, proportional_coeff: bool
            Whether to expect a constant or proportional coefficient.
        break_off_real_poles: bool
            Whether to group poles and residues by realness of poles.

        Returns
        -------
        dict
            with keys {'poles', 'residues'} and optionally also
            {'constants', 'proportionals'}. If `break_off_real_poles` is
            true, the values for the keys {'poles', 'residues'} are
            themselves dicts with keys {'real', 'complex'}. In all cases
            the ordering of the poles is the same as that of the residues.
        '''
        norm_val = self._get_norm(norm)

        residues_idx = n_poles_real + 2 * n_poles_cmplx
        residues_end_idx = 2 * residues_idx

        expected_nargs = residues_end_idx \
            + int(constant_coeff) + int(proportional_coeff)

        def fit_param_raveler(args):
            args = np.asarray(args)
            if len(args) != expected_nargs:
                raise ValueError

            def split_real_imag(arr):
                if len(arr) != residues_idx:
                    raise ValueError

                return (
                    arr[:n_poles_real],
                    arr[n_poles_real:n_poles_real+n_poles_cmplx],
                    arr[n_poles_real+n_poles_cmplx:],
                )

            poles_real, poles_cmplx_real, poles_cmplx_imag = \
                split_real_imag(norm_val * args[:residues_idx])
            residues_real, residues_cmplx_real, residues_cmplx_imag = \
                split_real_imag(
                    norm_val * args[residues_idx:residues_end_idx],
                )
            misc_terms_normed = iter(args[residues_end_idx:])

            retval = {
                'poles': {
                    'real': {
                        'real': poles_real,
                        'imag': 0 * poles_real,
                        # like zeros-like, but also handles case
                        # where poles_real contains uncertain values
                    },
                    'complex': {
                        'real': poles_cmplx_real,
                        'imag': poles_cmplx_imag,
                    },
                },
                'residues': {
                    'real': {
                        'real': residues_real,
                        'imag': 0 * residues_real,
                    },
                    'complex': {
                        'real': residues_cmplx_real,
                        'imag': residues_cmplx_imag,
                    },
                },
            }
            if constant_coeff:
                retval['constants'] = next(misc_terms_normed)
            if proportional_coeff:
                retval['proportionals'] = next(misc_terms_normed) / norm_val

            try:
                next(misc_terms_normed)
                raise ValueError('Too many arguments!')
            except StopIteration:
                pass

            # casework through all four combos of `complex_values` and `break_off_real_poles`

            if not complex_values and break_off_real_poles:
                return retval
            elif not complex_values and not break_off_real_poles:
                for key in ['poles', 'residues']:
                    retval[key] = {
                        subsubkey: np.concatenate((
                            retval[key]['real'][subsubkey],
                            retval[key]['complex'][subsubkey],
                        ))
                        for subsubkey in ['real', 'imag']
                    }
                return retval

            assert complex_values

            for key in ['poles', 'residues']:
                for subkey in ['real', 'complex']:
                    values_dict = retval[key][subkey]
                    retval[key][subkey] = values_dict['real'] + 1j * values_dict['imag']

            if not break_off_real_poles:
                for key in ['poles', 'residues']:
                    retval[key] = np.hstack((
                        retval[key]['real'],
                        retval[key]['complex'],
                    ))
            return retval

        return fit_param_raveler

    def make_fit_function(
            self,
            n_poles_real: int,
            n_poles_cmplx: int,
            constant_coeff: bool,
            proportional_coeff: bool,
            norm: bool | float = True):

        raveler = self.make_fit_param_raveler(
            n_poles_real,
            n_poles_cmplx,
            constant_coeff,
            proportional_coeff,
            break_off_real_poles=True,
            complex_values=True,
            norm=False,  # work with normalized values in the fit function
        )

        def fit_function(x, *args):
            '''
            Parameters
            ----------
            x : arraylike (2,) or (N, 2)
                Pair (or N pairs) of fit function inputs: (frequency, realimag_flag)
                where realimag_flag is (0, 1)
            args
                Unraveled fit function parameters, ordered as defined in
                `unravel_fit_params` and normalized as specified by `norm`
            '''
            x = np.asarray(x)
            freq, realimag = x[..., 0], x[..., 1]

            # create extra dimension to be broadcast over n_poles
            s = 2 * np.pi * 1j * freq.reshape(-1, 1) / self._get_norm(norm)

            params = raveler(args)

            realpole_contrib = np.sum(
                params['residues']['real'] / (s - params['poles']['real']),
                axis=-1,
            )
            cmplxpole_contrib = np.sum(
                params['residues']['complex'] / (s - params['poles']['complex'])
                + (
                    np.conj(params['residues']['complex'])
                    /
                    (s - np.conj(params['poles']['complex']))
                ),
                axis=-1,
            )

            retval_complex = realpole_contrib + cmplxpole_contrib
            if constant_coeff:
                retval_complex += params['constants']
            if proportional_coeff:
                retval_complex += params['proportional'] * s

            # np.real if realimag == 0 else np.imag
            factor = np.ones_like(realimag, dtype='complex')
            factor[(realimag == 1)] = 1j
            return np.real(retval_complex / factor)

        def jacobian(x, *args):
            '''
            Parameters
            ----------
            x : array_like, (..., 2)
                Tuples (frequency [Hz], {0, 1}) where the second term specifies
                whether to specify real and imaginary parts.
            args : float
                Unraveled fit function parameters, ordered as in
                `unravel_fit_params` and normalized as specified by `norm`.

            Returns
            -------
            array_like, (..., n_poles)
                The real or imaginary parts of the Jacobian evaluated at the
                frequencies in `x`. The choice of real or imagainary part is
                determined by the {0, 1} values in `x`.
            '''
            x = np.asarray(x)
            freq, realimag = x[..., 0], x[..., 1]

            # create extra dimension to be broadcast over n_poles
            s = 2 * np.pi * 1j * freq[..., np.newaxis] / self._get_norm(norm)

            params = raveler(args)

            realpole_grad = params['residues']['real'] / (s - params['poles']['real'])**2
            cmplxpole_realpart_grad = (
                params['residues']['complex'] / (s - params['poles']['complex'])**2
                + (
                    np.conj(params['residues']['complex'])
                    /
                    (s - np.conj(params['poles']['complex']))**2
                )
            )
            cmplxpole_imagpart_grad = 1j * (
                params['residues']['complex'] / (s - params['poles']['complex'])**2
                - (
                    np.conj(params['residues']['complex'])
                    /
                    (s - np.conj(params['poles']['complex']))**2
                )
            )

            realpole_res_realpart_grad = 1 / (s - params['poles']['real'])
            cmplxpole_res_realpart_grad = (
                1 / (s - params['poles']['complex'])
                + 1 / (s - np.conj(params['poles']['complex']))
            )

            # realpole_res_imagpart_grad = 1j * realpole_res_realpart_grad
            cmplxpole_res_imagpart_grad = 1j * (
                1 / (s - params['poles']['complex'])
                - 1 / (s - np.conj(params['poles']['complex']))
            )

            grad_cmpnts = [
                realpole_grad,
                cmplxpole_realpart_grad,
                cmplxpole_imagpart_grad,
                realpole_res_realpart_grad,
                cmplxpole_res_realpart_grad,
                # realpole_res_imagpart_grad,
                cmplxpole_res_imagpart_grad,
            ]
            if constant_coeff:
                grad_cmpnts.append(np.ones_like(s))
            if proportional_coeff:
                grad_cmpnts.append(s)

            cmplx_gradient = np.concatenate(grad_cmpnts, axis=-1)  # shape: (n_freqs, n_poles)

            # np.real if realimag == 0 else np.imag
            factor = np.ones_like(realimag, dtype='complex')
            factor[(realimag == 1)] = 1j
            return np.real(cmplx_gradient / factor[:, np.newaxis])

        return fit_function, jacobian


def fit_network(network: rf.Network, n_poles_cmplx: Optional[int] = None):
    vf = VectorFittingFancy(network)
    if n_poles_cmplx is None:
        vf.auto_fit()
    else:
        vf.vector_fit(n_poles_real=0, n_poles_cmplx=n_poles_cmplx)
    vf.refine_fit()
    return vf


def fit_narrow_mode(
        network: rf.Network,
        n_poles_cmplx: Optional[int] = None,
        frequency_err_max: float = 400e+3):
    try:
        vf = fit_network(network, n_poles_cmplx)
        return vf.closest_pole_uparams(network.frequency.center, frequency_err_max)
    except RuntimeError:
        return (ufloat(np.nan, np.nan),) * 4


def fit_mode(network, center_ghz, span_ghz, **vf_kwargs):
    subnet = network._subnetwork(center_ghz, span_ghz)
    return fit_network(subnet, **vf_kwargs)


def test_a_fit(network, center_ghz, span_ghz, **vf_kwargs):
    vf = fit_mode(network, center_ghz, span_ghz, **vf_kwargs)
    vf.print_refined_poles()
    vf.visualize()
    return vf
