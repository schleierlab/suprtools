import re
from collections.abc import Callable, Sequence
from typing import Any, Unpack, assert_never, cast

import matplotlib
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure, SubFigure
from matplotlib.layout_engine import ConstrainedLayoutEngine
from matplotlib.ticker import AutoMinorLocator
from numpy.typing import ArrayLike

from sslab_txz.typing import AnnotateKwargs

from ._angleannotation import AngleAnnotation as AngleAnnotation
from .style import annotation_arrowprops_default


def minor_ticks_on(ax, which='both'):
    if ax.get_xscale() != 'log' and which in ['both', 'x']:
        ax.xaxis.set_minor_locator(AutoMinorLocator())
    if ax.get_yscale() != 'log' and which in ['both', 'y']:
        ax.yaxis.set_minor_locator(AutoMinorLocator())


def make_grids(ax):
    ax.grid(which='major', color='0.7')
    ax.grid(which='minor', color='0.9')


def remove_figure_padding(fig: Figure):
    pad_inches = matplotlib.rcParams['axes.linewidth'] / 2 / 72
    layout_engine = fig.get_layout_engine()
    if layout_engine is None or not isinstance(layout_engine, ConstrainedLayoutEngine):
        raise ValueError('Figure must be in constrained layout')
    layout_engine.set(
        w_pad=pad_inches,
        h_pad=pad_inches,
    )


def savefig_tightly(fig: Figure, fname: str, **kwargs):
    fig.savefig(
        fname,
        bbox_inches='tight',
        pad_inches='layout',
        **kwargs,
    )


def set_reci_ax(ax, invert: bool = False):
    sign = -1 if invert else +1
    def reci(x):
        return sign / (1e-15 + x)
    ax.set_xscale('function', functions=(reci, reci))


def sslab_style(ax):
    minor_ticks_on(ax)
    make_grids(ax)


def _digit_to_roman(digit: int, place: int) -> str:
    if not 1 <= digit <= 9:
        raise ValueError

    ROMAN = [('I', 'V'), ('X', 'L'), ('C', 'D'), ('M', 'MMMM')]
    one, five = ROMAN[place]
    ten, _ = ROMAN[place + 1]

    if digit == 4:
        return one + five
    elif digit == 9:
        return one + ten

    n_fives, n_ones = divmod(digit, 5)
    return five * n_fives + one * n_ones


def to_roman(n: int) -> str:
    digits_reversed = map(int, reversed(str(n)))
    return ''.join(reversed([_digit_to_roman(d, place) for place, d in enumerate(digits_reversed)]))


def expand_range(values: ArrayLike, factor: float = 1.1) -> tuple[np.float_, np.float_]:
    '''
    Given a set of real values with extrema val_min, val_max,
    give a pair of numbers lo, hi such that the interval [lo, hi]
    is centered on [val_min, val_max] and is longer by `factor`.
    '''
    values = np.asarray(values)
    halfspan = (values.max() - values.min()) / 2
    midpt = (values.max() + values.min()) / 2
    return midpt - halfspan * factor, midpt + halfspan * factor


def frexp10(x):
    exp = int(np.floor(np.log10(abs(x))))
    return x / 10**exp, exp


def latex_frexp10(x):
    significand, exp = frexp10(x)

    if significand == 1:
        return fr'10^{{{exp}}}'
    return fr'{significand} \times 10^{{{exp}}}'


# TODO use AnchoredText instead of ax.text?
def label_subplots(
        fig: Figure,
        artists: Sequence[Axes] | Sequence[SubFigure],
        label_fmt='(alph)',
        adjust=(5, -5),
        colors=None,
        text_kws: dict[int, dict[str, Any]] = {},
):
    '''
    text_kws:
        Dictionary mapping subplot index to a dict of kwargs to pass to
        Axes.text() or SubFigre.text().
    '''
    if colors is None:
        colors = ['black'] * len(artists)

    for i, artist in enumerate(artists):
        # label physical distance in and down:
        adjust_x, adjust_y = adjust
        trans = matplotlib.transforms.ScaledTranslation(adjust_x/72, adjust_y/72, fig.dpi_scale_trans)
        # label = f'({chr(97 + i)})'

        def numsubber(m: re.Match[str]) -> str:
            match m.group(0):
                case 'alph':
                    return chr(97 + i)
                case 'Alph':
                    return chr(65 + i)
                case 'arabic':
                    return str(i + 1)
                case 'roman':
                    return to_roman(i + 1).lower()
                case 'Roman':
                    return to_roman(i + 1)
                case str():
                    raise ValueError
                case _:
                    raise TypeError

        label = re.sub(
            'alph|Alph|roman|Roman|arabic',
            numsubber,
            label_fmt,
        )

        artist = cast(Axes | SubFigure, artist)
        match artist:
            case Axes() as ax:
                transform = ax.transAxes
                text = ax.text
            case SubFigure() as subfig:
                transform = subfig.transSubfigure
                text = subfig.text
            case _:
                assert_never()

        text_kws_default = dict(
            transform=(transform + trans),
            fontsize='large',
            verticalalignment='top',
            color=colors[i],
        )
        text(
            0.0, 1.0,
            label,
            **(text_kws_default | text_kws.get(i, dict())),
            # fontfamily='serif',
        )


def watermark(ax: Axes, text: str, **kwargs):
    default_kw = dict(
        color='red',
        alpha=0.5,
        fontsize=30,
        ha='center',
        va='center',
        rotation=30,
    )
    ax.text(
        0.5, 0.5,
        text,
        transform=ax.transAxes,
        fontdict=None,  # redundant line for mypy
        **(default_kw | kwargs),
    )


def annotate_length(
        ax,
        text,
        left_endpt,
        right_endpt,
        offset_points=8,
        horizontalalignment='center',
        verticalalignment='center',
        reverse=False,
        arrowprops=dict(),
):
    left_endpt = np.asarray(left_endpt)
    right_endpt = np.asarray(right_endpt)

    midpt = (left_endpt + right_endpt) / 2
    ax.annotate(
        '',
        xy=left_endpt,
        xycoords=ax.transData,
        xytext=right_endpt,
        textcoords=ax.transData,
        arrowprops=(annotation_arrowprops_default | dict(arrowstyle='<|-|>') | arrowprops),
    )

    linevec_x, linevec_y = right_endpt - left_endpt
    normalvec = np.array([linevec_y, -linevec_x])
    normalvec /= np.linalg.norm(normalvec)
    if normalvec[1] < 0:
        normalvec = -normalvec
    elif normalvec[1] == 0 and normalvec[0] < 0:
        normalvec = -normalvec

    ax.annotate(
        text,
        xy=midpt,
        xycoords=ax.transData,
        xytext=normalvec * offset_points * (-1 if reverse else +1),
        textcoords='offset points',
        horizontalalignment=horizontalalignment,
        verticalalignment=verticalalignment,
    )


def annotate_radius(
        ax: Axes, text: str,
        center_xy, radius, annotation_angle, annotation_distance,
        arrowprops=dict()):
    center_xy = np.asarray(center_xy)

    unit_vec = np.array([np.cos(annotation_angle), np.sin(annotation_angle)])
    annotation_xy = center_xy + radius * unit_vec
    text_xy = center_xy + (radius + annotation_distance) * unit_vec

    ax.annotate(
        text,
        xy=annotation_xy,
        xycoords='data',
        xytext=text_xy,
        textcoords='data',
        arrowprops=(annotation_arrowprops_default | dict(arrowstyle='-|>') | arrowprops)
    )


def annotate_line(
        ax: Axes,
        text: str,
        annotation_x: float,
        line_func: Callable[[ArrayLike], ArrayLike],
        offset_pts: tuple[float, float] = (0, 0),
        **kwargs: Unpack[AnnotateKwargs],
):
    test_xs = annotation_x + np.array([0, 0.1])
    test_ys = line_func(test_xs)
    plot_xys = np.transpose([test_xs, test_ys])

    # rotation_angle = np.rad2deg(np.arctan2(*(plot_xys[1] - plot_xys[0])[::-1]))
    rotation_angle = ax.transData.transform_angles(
        np.rad2deg(np.arctan2(*(plot_xys[1] - plot_xys[0])[::-1])).reshape(-1),
        plot_xys[0].reshape(-1, 2),
    )[0]
    ax.annotate(
        text,
        xy=plot_xys[0],
        xycoords='data',
        xytext=offset_pts,
        textcoords='offset points',
        rotation=rotation_angle,
        rotation_mode='anchor',
        transform_rotates_text=True,
        **kwargs,
    )


def mpl_usetex() -> bool:
    return matplotlib.rcParams['text.usetex']
