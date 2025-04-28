from collections.abc import Callable
from typing import Any, Literal

import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import Colormap, LinearSegmentedColormap
from seaborn.miscplot import palplot
from seaborn.palettes import color_palette

# copied from seaborn widgets.py
try:
    from ipywidgets import FloatSlider, IntSlider, interact
except ImportError:
    def interact(f):
        msg = "Interactive palettes require `ipywidgets`, which is not installed."
        raise ImportError(msg)


# copied from seaborn palettes.py
class _ColorPalette(list):
    """Set the color palette in a with statement, otherwise be a list."""
    def __enter__(self):
        """Open the context."""
        from seaborn.rcmod import set_palette
        self._orig_palette = color_palette()
        set_palette(self)
        return self

    def __exit__(self, *args):
        """Close the context."""
        from seaborn.rcmod import set_palette
        set_palette(self._orig_palette)

    def as_hex(self):
        """Return a color palette with hex codes instead of RGB values."""
        hex = [mpl.colors.rgb2hex(rgb) for rgb in self]
        return _ColorPalette(hex)

    def _repr_html_(self):
        """Rich display of the color palette in an HTML frontend."""
        s = 55
        n = len(self)
        html = f'<svg  width="{n * s}" height="{s}">'
        for i, c in enumerate(self.as_hex()):
            html += (
                f'<rect x="{i * s}" y="0" width="{s}" height="{s}" style="fill:{c};'
                'stroke-width:2;stroke:rgb(255,255,255)"/>'
            )
        html += '</svg>'
        return html


def cubehelix_doublegamma_palette(
        n_colors: int = 6,
        start: float = 0,
        rot: float = 0.4,
        gamma: float = 1.0,
        gamma_rot: float = 1.0,
        hue: float = 0.8,
        light: float = 0.85,
        dark: float = 0.15,
        reverse: bool = False,
        as_cmap: bool = False,
):
    """"""
    # Copied from seaborn
    def get_color_function(p0: float, p1: float) -> Callable[[float], float]:
        # Copied from matplotlib because it lives in private module
        def color(x: float) -> float:
            # Apply gamma factor to emphasise low or high intensity values
            xg = x ** gamma

            # Calculate amplitude and angle of deviation from the black
            # to white diagonal in the plane of constant
            # perceived intensity.
            a = hue * xg * (1 - xg) / 2

            phi = 2 * np.pi * (start / 3 + rot * x ** gamma_rot)

            return xg + a * (p0 * np.cos(phi) + p1 * np.sin(phi))
        return color

    cdict: dict[Literal['red', 'green', 'blue'], Any] = {
        "red": get_color_function(-0.14861, 1.78277),
        "green": get_color_function(-0.29227, -0.90649),
        "blue": get_color_function(1.97294, 0.0),
    }

    cmap: Colormap
    cmap = mpl.colors.LinearSegmentedColormap("cubehelix_doublegamma", cdict)  # type: ignore

    x = np.linspace(light, dark, int(n_colors))
    pal = cmap(x)[:, :3].tolist()
    if reverse:
        pal = pal[::-1]

    if as_cmap:
        x_256 = np.linspace(light, dark, 256)
        if reverse:
            x_256 = x_256[::-1]
        pal_256 = cmap(x_256)
        cmap = mpl.colors.ListedColormap(pal_256, "seaborn_cubehelix")
        return cmap
    else:
        return _ColorPalette(pal)


# WIDGETS

def _init_mutable_colormap():
    """
    Create a matplotlib colormap that will be updated by the widgets.
    Copied from seaborn.
    """
    greys = color_palette("Greys", 256)
    cmap = LinearSegmentedColormap.from_list("interactive", greys)
    cmap._init()
    cmap._set_extremes()
    return cmap


def _update_lut(cmap, colors):
    """Change the LUT values in a matplotlib colormap in-place."""
    cmap._lut[:256] = colors
    cmap._set_extremes()


def _show_cmap(cmap):
    """Show a continuous matplotlib colormap."""
    from seaborn.rcmod import axes_style  # Avoid circular import
    with axes_style("white"):
        f, ax = plt.subplots(figsize=(8.25, .75))
    ax.set(xticks=[], yticks=[])
    x = np.linspace(0, 1, 256)[np.newaxis, :]
    ax.pcolormesh(x, cmap=cmap)

def choose_cubehelix_doublegamma_palette(as_cmap=False):
    """Launch an interactive widget to create a sequential cubehelix palette.

    This corresponds with the :func:`cubehelix_palette` function. This kind
    of palette is good for data that range between relatively uninteresting
    low values and interesting high values. The cubehelix system allows the
    palette to have more hue variance across the range, which can be helpful
    for distinguishing a wider range of values.

    Requires IPython 2+ and must be used in the notebook.

    Parameters
    ----------
    as_cmap : bool
        If True, the return value is a matplotlib colormap rather than a
        list of discrete colors.

    Returns
    -------
    pal or cmap : list of colors or matplotlib colormap
        Object that can be passed to plotting functions.

    See Also
    --------
    cubehelix_palette : Create a sequential palette or colormap using the
                        cubehelix system.

    """
    pal = []
    if as_cmap:
        cmap = _init_mutable_colormap()

    @interact
    def choose_cubehelix_doublegamma(n_colors=IntSlider(min=2, max=16, value=9),
                         start=FloatSlider(min=0, max=3, value=0),
                         rot=FloatSlider(min=-3, max=3, value=.4),
                         gamma=FloatSlider(min=0, max=5, value=1),
                         gamma_rot=FloatSlider(min=0, max=10, value=1),
                         hue=FloatSlider(min=0, max=1, value=.8),
                         light=FloatSlider(min=0, max=1, value=.85, step=0.01),
                         dark=FloatSlider(min=0, max=1, value=.15, step=0.01),
                         reverse=False):

        if as_cmap:
            colors = cubehelix_doublegamma_palette(
                256, start, rot, gamma, gamma_rot,
                                       hue, light, dark, reverse)
            _update_lut(cmap, np.c_[colors, np.ones(256)])
            _show_cmap(cmap)
        else:
            pal[:] = cubehelix_doublegamma_palette(
                n_colors, start, rot, gamma, gamma_rot,
                                       hue, light, dark, reverse)
            palplot(pal)

    if as_cmap:
        return cmap
    return pal
