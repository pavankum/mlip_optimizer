"""Shared type definitions for the mlip_optimizer package."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TypeAlias

Color: TypeAlias = Iterable[float]
BondIndices: TypeAlias = tuple[int, int]
AngleIndices: TypeAlias = tuple[int, int, int]
TorsionIndices: TypeAlias = tuple[int, int, int, int]

BondLengths: TypeAlias = dict[BondIndices, float]
BondAngles: TypeAlias = dict[AngleIndices, float]
TorsionAngles: TypeAlias = dict[TorsionIndices, float]
