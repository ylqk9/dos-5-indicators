from pathlib import Path

import numpy as np
import rtoml as toml

from common import get_dbc_with_width, get_width, load_dos
from pkm import Model


def run(in_path: Path, out_path: str | Path):
    in_path = Path(in_path)
    out_path = Path(out_path)
    data = load_dos(in_path)

    dbc_ans = {}
    width_ans = {}
    nonbonding_ans = {}
    
    for el, ed in data.items():
        dos = np.array(ed["dos"])
        energy = np.array(ed["energy"])
        
        dosF, energyF = Model().run(dos, energy)

        dbc = get_dbc_with_width(dosF, energyF, dx=energy[1] - energy[0])
        dbc_ans[el] = float(dbc)
        width = get_width(dosF, energyF)
        width_ans[el] = float(width)
        mask = (~np.isin(energy, energyF)) & ((energy <= energyF.max()) & (energy >= energyF.min()))
        nonbonding_dbc = get_dbc_with_width(dos[mask], energy[mask], dx=energy[1] - energy[0])

        nonbonding_ans[el] = float(nonbonding_dbc)

    ans = {"dbc": dbc_ans, "width": width_ans, "nonbonding": nonbonding_ans}    
    toml.dump(ans, out_path)

if __name__ == "__main__": 
    run("dos.toml", "basis.toml")
