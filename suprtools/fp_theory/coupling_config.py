from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar


@dataclass
class CouplingConfig:
    no_xcoupling: ClassVar[CouplingConfig]
    config_nopolmix: ClassVar[CouplingConfig]
    all_xcoupling: ClassVar[CouplingConfig]
    paraxial: ClassVar[CouplingConfig]

    prop: bool = True
    prop_xcoupling: bool = False
    wave: bool = True
    wave_xcoupling: bool = False
    vec: bool = True
    vec_xcoupling: bool = False
    astig: bool = True
    astig_xcoupling: bool = False
    asphere: bool = True
    asphere_xcoupling: bool = False
    v_plus_a: bool = True


CouplingConfig.no_xcoupling = CouplingConfig()
CouplingConfig.config_nopolmix = CouplingConfig(v_plus_a=False)
CouplingConfig.all_xcoupling = CouplingConfig(
    prop_xcoupling=True,
    wave_xcoupling=True,
    vec_xcoupling=True,
    astig_xcoupling=True,
    asphere_xcoupling=True,
)
CouplingConfig.paraxial = CouplingConfig(
    prop=False,
    wave=False,
    vec=False,
    astig=True,
    asphere=False,
)
