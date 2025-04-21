import tomllib
from pathlib import Path
from typing import Literal

import numpy as np
from numpy import ndarray


def load_dos(path: Path) -> dict[str, dict[Literal["dos", "energy"], list[float]]]:
    path = Path(path) 
    with path.open("rb") as _:
        data = tomllib.load(_)
    return data

def get_dbc(dos: ndarray, energy: ndarray):
    n = dos.shape[0]
    if n == 1:
        return energy[0]
    dos = dos.copy()
    energy = energy.copy()
    index = np.argsort(energy)
    energy = energy[index]
    dos = dos[index]
    div = (energy[1:] - energy[:-1]) / 2
    
    left = np.zeros(n)
    left[1:] = div 
    left[0] = left[1]

    right = np.zeros(n)
    right[:-1] = div 
    right[-1] = right[-2]

    A = np.sum((left + right) * dos * energy)
    B = np.sum((left + right) * dos)

    return A / B

def get_dbc_with_width(dos: ndarray, energy: ndarray, dx: float):
    A = np.sum(dx * dos * energy)
    B = np.sum(dx * dos)
    return A / B

def get_width(dos: np.ndarray, energy: np.ndarray):
    dos = dos.copy() 
    energy = energy.copy()
    index = np.argsort(energy)
    energy = energy[index]
    dos = dos[index]
    
    dbc = get_dbc(dos, energy)
    
    n = max(energy.shape)
    div = (energy[1:] - energy[:-1]) / 2
    
    left = np.zeros(n)
    left[1:] = div 
    left[0] = left[1]

    right = np.zeros(n)
    right[:-1] = div 
    right[-1] = right[-2]

    A = np.sum((left + right) * dos * ((energy - dbc) ** 2))
    B = np.sum((left + right) * dos)

    return np.sqrt(A / B)

def to_polar(dos: ndarray, energy: ndarray) -> tuple[ndarray, ndarray]:
    energy = np.pi * (energy - energy.min()) / (energy.max() - energy.min()) 
    dos = (dos) / (dos.max())
    return dos, energy

def polar_in_cartesian(r: ndarray, theta: ndarray) -> tuple[ndarray, ndarray]:
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y 
