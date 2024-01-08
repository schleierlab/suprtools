import datetime
import functools
from pathlib import Path
from typing import Literal

import h5py
import matplotlib.gridspec
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import skrf as rf
from matplotlib.ticker import MultipleLocator
from scipy.constants import pi
from tqdm import tqdm

from sslab_txz.plotting import sslab_style


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
        return scipy.signal.sosfilt(self.sos, data)

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
            fclp = r'f_{c, \mathrm{LP}}'
            return f'${fclp}{inv} = {1e-6/self.fc_lp:.3f}$ MHz'
        if self.low_pass_order == 0:
            fchp = r'f_{c, \mathrm{HP}}'
            return f'${fchp}{inv} = {1e-6/self.fc_hp:.3f}$ MHz'

        fchlp = r'f_{c, \mathrm{HP,LP}}'
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

        left_endpt = round(center_ghz - halfspan_ghz, ndigits=5)
        right_endpt = round(center_ghz + halfspan_ghz, ndigits=5)
        index_str = f'{left_endpt}-{right_endpt}ghz'

        return self[index_str]

    def plot_filtered_network(
        self,
        filt: ScanDataFilter,
        fig=None,
        axs=None,
    ):
        make_fig = fig is None or axs is None
        if make_fig:
            fig, axs = plt.subplots(figsize=(9, 3.6), constrained_layout=True, nrows=2, sharex=True)
        for ax in axs:
            sslab_style(ax)

        # zi = scipy.signal.sosfilt_zi(sos_filter)
        # filtered_network, _ = scipy.signal.sosfilt(sos_filter, network.s.flatten(), zi=zi)

        filtered_network = filt.filt(self.s.flatten())

        ax1, ax2 = axs
        ax1.plot(
            self.f / 1e+9,
            rf.complex_2_magnitude(self.s.flatten()),
            linewidth=0.75,
            label='Raw data',
        )
        ax2.plot(
            self.f / 1e+9,
            rf.complex_2_magnitude(filtered_network),
            linewidth=0.25,
            label=fr'Filtered, {filt.latex_repr()}',
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
            fsr_guess_ghz,
            offset_range,
            scale=1,
            filt=None,
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
                **kwargs,
            )

        if filt is not None:
            filtered_axs = axs[1::2]
            for ax, offset in zip(filtered_axs, offset_iter):
                offset_center_freq_ghz = center_freq_ghz + offset * fsr_guess_ghz
                subnet = self._subnetwork(offset_center_freq_ghz, span_ghz)

                filtered_magnitude = \
                    rf.complex_2_magnitude(filt.filt(subnet.s[:, 0, 0] * scale))[200:]
                # noise_floor = np.quantile(filtered_magnitude, 0.99)
                # peak_inds, _ = scipy.signal.find_peaks(
                #     filtered_magnitude,
                #     prominence=2*noise_floor,
                #     distance=int(2e6//self.freq_step),
                # )
                ax.plot(
                    subnet.f[200:] / 1e9 - fsr_guess_ghz * offset,
                    filtered_magnitude,
                    label=fr'${offset:+d}\times {fsr_guess_ghz}$ GHz',
                    **kwargs,
                )
                # ax.plot(
                #     subnet.f[200:][peak_inds] / 1e9 - fsr_guess_ghz * offset,
                #     filtered_magnitude[peak_inds],
                #     marker='x',
                #     color='C1',
                #     linestyle='None',
                # )

            for ax_db, ax_filt, offset in zip(unfiltered_axs, filtered_axs, offset_iter):
                ax_db.set_ylabel(f'$|S_{{21}}|$ (dB)\n[+${offset}\\times$ FSR]')
                ax_filt.set_ylabel("filtered\n(linear)")

        for ax in axs:
            sslab_style(ax)

        fig.supxlabel('Frequency (GHz)')
        if filt is None:
            fig.supylabel('$S_{21}$ (dB)')

        return fig, axs


class WideScanData():
    def __init__(self, s11, s21, metadata):
        self.s11 = s11
        self.s21 = s21
        self.metadata = metadata

    @staticmethod
    def _get_s_param(f: h5py.File, param: Literal['s11', 's21']):
        datapath = f'data/s_params/{param}'
        real_part = np.array(f.get(f'{datapath}/real'))
        imag_part = np.array(f.get(f'{datapath}/imag'))
        return real_part + 1j*imag_part

    @classmethod
    def _load_window_data(cls, f, sample_temps=False):
        frequencies_dataset = f.get('data/frequencies')
        freq_obj = rf.Frequency.from_f(frequencies_dataset, unit='Hz')
        s11_arr = cls._get_s_param(f, 's11')
        s21_arr = cls._get_s_param(f, 's21')

        end_time_str = f.attrs['run_time']
        end_datetime = datetime.datetime.strptime(end_time_str, '%Y%m%dT%H%M%S') \
            .replace(tzinfo=datetime.timezone.utc)

        if sample_temps:
            metadata_pt = (
                f.attrs['temperature_still'],
                f.attrs['temperature_sample'],
                end_datetime,
            )
        else:
            metadata_pt = (
                f.attrs['temperature_still'],
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
            names = ['t_still', 't_samp', 'end_time']
        else:
            names = ['t_still', 'end_time']

        metadata_arr = np.rec.fromrecords([metadata_pt], names=names)
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
        stitched_s11 = functools.reduce(rf.network.stitch, (d.s11 for d in wide_scan_datas))
        stitched_s21 = functools.reduce(rf.network.stitch, (d.s21 for d in wide_scan_datas))
        stitched_s11.drop_non_monotonic_increasing()
        stitched_s21.drop_non_monotonic_increasing()

        return cls(s11=stitched_s11, s21=stitched_s21, metadata=metadata)

    @classmethod
    def from_windows(
        cls,
        h5s_path: Path,
        network_name: str,
        sample_temps: bool = False,
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
                    cls._load_window_data(f, sample_temps=sample_temps)
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
            names = ['t_still', 't_samp', 'end_time']
        else:
            names = ['t_still', 'end_time']

        metadata_arr = np.rec.fromrecords(metadata, names=names)
        return cls(s11=s11_net, s21=s21_net, metadata=metadata_arr)

    def __add__(self, other):
        s11 = WideScanNetwork(rf.stitch(self.s11, other.s11))
        s21 = WideScanNetwork(rf.stitch(self.s21, other.s21))
        s11.drop_non_monotonic_increasing()
        s21.drop_non_monotonic_increasing()

        metadata = np.concatenate((self.metadata, other.metadata))
        return WideScanData(s11, s21, metadata)


class VectorFittingFancy(rf.VectorFitting):
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
        return -2 * np.real(self.poles) / (2 * pi)

    @property
    def resonances(self):
        return np.imag(self.poles) / (2 * pi)

    def print_poles(self, freq_prec=3, fwhm_prec=4):
        print('Freq (GHz)\tFWHM (MHz)')
        for pole in sorted(self.poles, key=np.imag):
            pole_freq_ghz = np.imag(pole) / (2 * pi * 1e9)
            pole_freq_fmt_str = f'{{:5.{freq_prec}f}}'
            pole_freq_str = pole_freq_fmt_str.format(pole_freq_ghz)
            pole_fwhm_fmt_str = f'{{:9.{fwhm_prec}f}}'
            pole_fwhm_str = pole_fwhm_fmt_str.format(-2 * np.real(pole) / (2 * pi * 1e6))
            print(f'{pole_freq_str}\t\t{pole_fwhm_str}')

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

        ax_iq.set_title(fr'$S_{{21}} \times {polar_scale_latex_str}$')

        self._setup_iq_ax(ax_iq_zoom, polar_scale_latex_str)

        # xoffset = int(freq_obj.center / freq_obj.multiplier) * freq_obj.multiplier
        # def freq_to_xunit(f):
        #     return (f - xoffset) / freq_obj.multiplier_dict[plot_unit]

        network = self.network
        freq_obj = network.frequency
        freq_min, freq_max = freq_obj.f[[0, -1]]
        freq_range = np.linspace(freq_min, freq_max, 5001)

        if plot_unit is None:
            if freq_obj.span > 1e+6:
                plot_unit = 'mhz'
            elif freq_obj.span > 1e+3:
                plot_unit = 'khz'
            elif freq_obj.span > 1e+0:
                plot_unit = 'hz'

        xoffset, offset_multiplier, freq_to_xunit = self._get_offset_and_func(
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

        # offset_data_freqs = freq_to_xunit(freq_obj.f)
        offset_freq_range = freq_to_xunit(freq_range)

        model_response = self.get_model_response(0, 0, freqs=freq_range)
        model_response_discrete = self.get_model_response(0, 0, freqs=network.frequency.f)

        marker_plot_kwargs = dict(
            marker='.',
            markersize=1,
            linestyle='None',
        )

        model_plot_kwargs = dict(
            linewidth=1.5,
            alpha=0.5,
            color='red',
        )

        lines = ax_mag.plot(
            freq_to_xunit(network.frequency.f),
            rf.mathFunctions.complex_2_db(network.s.flatten()),
            **marker_plot_kwargs,
        )
        # line = lines[0]
        ax_mag.plot(
            offset_freq_range,
            rf.mathFunctions.complex_2_db(model_response),
            **model_plot_kwargs,
        )

        ax_phase.plot(
            freq_to_xunit(network.frequency.f),
            np.unwrap(rf.mathFunctions.complex_2_degree(network.s.flatten()), period=360),
            **marker_plot_kwargs,
        )

        ax_phase.plot(
            offset_freq_range,
            np.unwrap(rf.mathFunctions.complex_2_degree(model_response), period=360),
            **model_plot_kwargs,
        )

        rms_error = np.sqrt(np.mean(np.abs((model_response_discrete - network.s.flatten())**2)))
        print(f'Mean sq error: {rms_error:.3E}')

        scaled_model_response = model_response_discrete * polar_scale
        scaled_sparam = network.s.flatten() * polar_scale

        iq_model_plot_kw = dict(
            marker='x',
            linestyle='None',
            markersize=1,
            color='red',
        )
        ax_iq.plot(
            rf.complex_2_radian(scaled_model_response),
            rf.complex_2_magnitude(scaled_model_response),
            # color=lines[0].get_color(),
            **iq_model_plot_kw,
        )

        ax_iq_zoom.plot(
            np.real(scaled_model_response),
            np.imag(scaled_model_response),
            # color=lines[0].get_color(),
            **iq_model_plot_kw,
        )

        iq_data_plot_kw = dict(
            marker='.',
            markersize=1,
            alpha=0.3,
            # linestyle='None',
            linewidth=1,
        )
        ax_iq.plot(
            rf.complex_2_radian(scaled_sparam),
            rf.complex_2_magnitude(scaled_sparam),
            color=lines[0].get_color(),
            **iq_data_plot_kw,
        )

        ax_iq_zoom.plot(
            np.real(scaled_sparam),
            np.imag(scaled_sparam),
            color=lines[0].get_color(),
            **iq_data_plot_kw,
        )

        xmin, xmax = ax_iq_zoom.get_xlim()
        ymin, ymax = ax_iq_zoom.get_ylim()

        xcent, ycent = (xmin + xmax) / 2, (ymin + ymax)/2
        maxspan = max(xmax - xmin, ymax - ymin)
        halfmaxspan = maxspan / 2

        ax_iq_zoom.set_xlim(xcent - halfmaxspan, xcent + halfmaxspan)
        ax_iq_zoom.set_ylim(ycent - halfmaxspan, ycent + halfmaxspan)

        return fig


def fit_mode(network, center_ghz, span_ghz, **vf_kwargs):
    subnet = network._subnetwork(center_ghz, span_ghz)
    vf = VectorFittingFancy(subnet)
    vf.vector_fit(
        n_poles_real=0,
        **vf_kwargs,
    )
    return vf


def test_a_fit(network, center_ghz, span_ghz, **vf_kwargs):
    vf = fit_mode(network, center_ghz, span_ghz, **vf_kwargs)
    vf.print_poles(freq_prec=9)
    vf.visualize()
    return vf
