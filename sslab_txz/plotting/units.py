from enum import Enum

from . import mpl_usetex


class Units(Enum):
    HZ = 'Hz'
    KHZ = 'KHz'
    MHZ = 'MHz'
    GHZ = 'GHz'

    S = 's'
    MS = 'ms'
    US = 'us'
    NS = 'ns'

    M = 'm'
    MM = 'mm'
    UM = 'um'
    NM = 'nm'

    K = 'K'
    MK = 'mK'

    OHM = 'ohm'

    DB = 'dB'

    # very flawed for sure but if it works it works...
    def mplstr(self) -> str:
        if mpl_usetex():
            # if self == self.OHM:
            #     return '\si{\ohm}'
            # elif self.value[-3:] == 'ohm':
            #     raise NotImplementedError

            return Rf'\si{{\{self.value}}}'

        if self.value[-3:] == 'ohm':
            return self.value[:-3] + R'$\Omega$'
        if self.value[0] == 'u':
            return Rf'$\mu${self.value[1:]}'

        return self.value
