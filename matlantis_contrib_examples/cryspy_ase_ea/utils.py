import os
import subprocess
import time
import pathlib
import itertools
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple

from ase.spacegroup.symmetrize import FixSymmetry
from ase.io import read, write

from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.ext.matproj import MPRester

from pfp_api_client.pfp.calculators.ase_calculator import ASECalculator
from pfp_api_client.pfp.estimator import Estimator, EstimatorCalcMode

from matlantis_features.features.common.opt import LBFGSASEOptFeature

os.environ["MATLANTIS_PFP_MODEL_VERSION"] = "v4.0.0"
os.environ["MATLANTIS_PFP_CALC_MODE"] = "crystal_u0"

estimator = Estimator(calc_mode=EstimatorCalcMode.CRYSTAL_U0)
calculator = ASECalculator(estimator)
shift_energies = estimator.get_shift_energy_table()

def query(element:str):
    MP_API_KEY = os.environ["MP_API_KEY"]
    with MPRester(MP_API_KEY) as mpr:
        # list_of_available_fields = mpr.summary.available_fields
        # binary = mpr.summary.search(chemsys=elem1 + "-" + elem2,
        binary = mpr.summary.search(chemsys=element,
                                  fields=["material_id",
                                          "elements", 
                                          "nsites",
                                          "composition",
                                          "structure",
                                          'uncorrected_energy_per_atom',
                                          'energy_per_atom',
                                          'formation_energy_per_atom',
                                          'energy_above_hull',])

    return binary

def create_binary(elem1, elem2, composition):
    binary_lst = []
    binary = namedtuple('binary', ['nsites', 'composition', 'material_id'])
    for i in composition:
        binary_lst.append(binary(nsites=i[0] + i[1],
                                 composition={elem1:i[0], elem2:i[1]},
                                 material_id="Original composition was specified."))
    
    return binary_lst

def get_base_energy(ele: list):
    "Calculate energy for single element of Material Project and Matlantis"
    for i in ele:
        if i.energy_above_hull == 0.0:
            base_energy_pymatgen = i.energy_per_atom
            atoms = AseAtomsAdaptor.get_atoms(i.structure)
            total_shift_energy = sum([shift_energies[i] for i in atoms.get_atomic_numbers()])
            atoms.set_constraint([FixSymmetry(atoms)])
            opt = FireLBFGSASEOptFeature(filter=ExpCellASEFilter())
            result_opt = opt(atoms)
            total_energy = result_opt.atoms.ase_atoms.get_total_energy()
            base_energy_matlantis = (total_energy + total_shift_energy) / len(atoms)

            return base_energy_pymatgen, base_energy_matlantis

def make_cryspy_input(n, save_dir_path, target, mindist_elem1_elem1,
                      mindist_elem1_elem2, mindist_elem2_elem2, 
                      tot_struc, njob, n_crsov, n_perm, n_strain,
                      n_rand, n_elite, n_fittest, t_zie, maxgen_ea):
    e1, e2 = target.composition.items()
    natot = target.nsites
    os.makedirs(save_dir_path, exist_ok = True)
    os.makedirs(save_dir_path/"calc_in", exist_ok = True)
    
    cryspy_in = \
    f"""[basic]
algo = EA
calc_code = ASE
tot_struc = {tot_struc}
nstage = 1
njob = {njob}
jobcmd = bash
jobfile = job_cryspy

[structure]
natot = {natot}
atype = {str(e1[0])} {str(e2[0])}
nat = {int(e1[1])} {int(e2[1])}
mindist_1 = {mindist_elem1_elem1} {mindist_elem1_elem2}
mindist_2 = {mindist_elem1_elem2} {mindist_elem2_elem2}

[ASE]
ase_python = ase_in.py

[EA]
n_pop = {n_crsov+n_perm+n_strain+n_rand}
n_crsov = {n_crsov}
n_perm = {n_perm}
n_strain = {n_strain}
n_rand = {n_rand}
n_elite = {n_elite}
n_fittest = {n_fittest}
t_size = {t_size}
fit_reverse = False
slct_func = TNM
crs_lat = equal
nat_diff_tole = 4
ntimes = 1
maxcnt_ea = 50
maxgen_ea = {maxgen_ea}

[option]
"""

    job_cryspy =\
    """#!/bin/sh

# ---------- ASE
python ase_in.py

# ---------- CrySPY
sed -i -e '3 s/^.*$/done/' stat_job
    """

    ase_in =\
    """import os
import numpy as np

from ase.io import read, write
from ase.spacegroup.symmetrize import FixSymmetry

from matlantis_features.features.common.opt import LBFGSASEOptFeature

os.environ["MATLANTIS_PFP_MODEL_VERSION"] = "v4.0.0"
os.environ["MATLANTIS_PFP_CALC_MODE"] = "crystal_u0"

# ---------- input structure
# CrySPY outputs 'POSCAR' as an input file in work/xxxxxx directory
atoms = read('POSCAR', format='vasp')

# ---------- setting and run    
def filter_structure(atoms_cell):
    n = np.cross(atoms_cell[0], atoms_cell[1])
    c = atoms_cell[2]
    # Calculate the dot product between c and n
    dot_product = np.dot(c, n)

    # Calculate the magnitudes of c and n
    mag_c = np.linalg.norm(c)
    mag_n = np.linalg.norm(n)

    # Calculate the cosine of the angle theta
    cos_theta = dot_product / (mag_c * mag_n)

    # Calculate the angle theta in radians
    theta_rad = np.arccos(np.clip(cos_theta, -1, 1))

    # Convert theta from radians to degrees if needed
    theta_deg = np.degrees(theta_rad)

    # print("Theta (in degrees):", np.abs(90 - theta_deg))

    if np.abs(90 - theta_deg) > 25.0:
        return True
    else:
        return False

if filter_structure(atoms.cell):
    # atoms.set_constraint([FixSymmetry(atoms)])
    opt = LBFGSASEOptFeature(filter=True)
    result_opt = opt(atoms)
    e = result_opt.atoms.ase_atoms.get_total_energy()
    with open('log.tote', mode='w') as f:
        f.write(str(e))

    write('CONTCAR', result_opt.atoms.ase_atoms, format='vasp')

else:
    with open('log.tote', mode='w') as f:
        f.write(str(0.00))

    write('CONTCAR', atoms, format='vasp')

    """
    
    print(save_dir_path/"cryspy.in")
    with open(save_dir_path/"cryspy.in", "w") as f:
        f.write(cryspy_in)
    with open(save_dir_path/"calc_in"/"job_cryspy", "w") as f:
        f.write(job_cryspy)
    with open(save_dir_path/"calc_in"/"ase_in.py_1", "w") as f:
        f.write(ase_in)
    with open(save_dir_path/"material_id.txt", "w") as f:
        f.write(target.material_id)

def check_log_for_string(log_path, target_string):
    """Check if the target string exists in the log file."""
    with open(log_path, 'r') as file:
        return target_string in file.read()
        
def envelope(cryspy_energy:list, composition:list):
    data = cryspy_energy.copy()
    data.append(0)
    
    composition_ = composition.copy()
    composition_.append(1)
    env_x, env_y = [0], [0]
    f_fin = False
    for j in range(len(composition_) + 2):
        # initial
        env_dec = []
        env_inc = []
        if j == 0:
            x1 = 0
            y1 = 0
        # final
        elif j ==  (len(composition_) + 2) - 1:
            x1 = 1
            y1 = 0
        else:
            x1 = env_x[-1]
            y1 = env_y[-1]

        tmp_x2_list_dec, tmp_y2_list_dec = [], []
        tmp_x2_list_inc, tmp_y2_list_inc = [], []
        for i, x2 in enumerate(composition_):
            if x2 < x1:
                continue

            y2 = np.min(data[i])
            if x2 != x1:
                slope = (y2 - y1) / (x2 - x1)
            else:
                continue

            if y1 > y2:
                env_dec.append(np.abs(slope))
                tmp_x2_list_dec.append(x2)
                tmp_y2_list_dec.append(y2)
            elif y1 < y2:
                env_inc.append(np.abs(slope))
                tmp_x2_list_inc.append(x2)
                tmp_y2_list_inc.append(y2)

        if len(tmp_y2_list_dec) > 0:
            if y1 > min(tmp_y2_list_dec):
                dec_index = env_dec.index(max(env_dec))
                env_x.append(tmp_x2_list_dec[dec_index])
                env_y.append(tmp_y2_list_dec[dec_index])
            else:
                inc_index = env_inc.index(min(env_inc))
                env_x.append(tmp_x2_list_inc[inc_index])
                env_y.append(tmp_y2_list_inc[inc_index])
        elif len(tmp_y2_list_inc) > 0:
            if y1 < min(tmp_y2_list_inc):
                inc_index = env_inc.index(min(env_inc))
                env_x.append(tmp_x2_list_inc[inc_index])
                env_y.append(tmp_y2_list_inc[inc_index])
        else:
            env_x.append(1)
            env_y.append(0)
            f_fin = True

        if f_fin:
            break
            
    return env_x, env_y

def get_material_data(material_id_path, binary):
    if os.path.exists(material_id_path):
        with open(material_id_path) as f:
            material_id = f.read()
            
    idx = [str(i.material_id) for i in binary]
    mp_data = binary[idx.index(material_id)]
            
    return mp_data

def check_natom(cryspy_in_path, nat1, nat2):
    if os.path.exists(cryspy_in_path):
        with open(cryspy_in_path) as f:
            for line in f.read().splitlines():
                if "nat = " in line:
                    nat = line.split("=")[-1].strip()
                    break
    else:
        assert f"{cryspy_in_path} doesn't exist."
    
    if nat1 != int(nat.split()[0]) and nat2 != int(nat.split()[1]):
        raise ValueError(f"natom in cryspy {nat} does not match material project {nat1}, {nat2}")
        
def get_matlantis_energy(structure):

    atoms = AseAtomsAdaptor.get_atoms(structure)
    total_shift_energy = sum([shift_energies[i] for i in atoms.get_atomic_numbers()])
    
    atoms.calc = calculator
    total_energy = atoms.get_total_energy()
    
    # atoms.set_constraint([FixSymmetry(atoms)])
    # filter=ExpCellASEFilter()
    # opt = FireLBFGSASEOptFeature(filter=filter)
    # result_opt = opt(atoms)
    # total_energy = result_opt.atoms.ase_atoms.get_total_energy()
    
    return total_energy / len(atoms), total_shift_energy / len(atoms)

def get_cryspy_data(energy_path, total_shift_energy, base_energy_matlantis):
    if os.path.exists(energy_path):
        energy_lst = []
        print(f"Found file at {energy_path}")
        with open(energy_path, 'r') as file:
            for line in file.read().splitlines():
                if len(line.split()) == 9:
                    energy = float(line.split()[6]) + total_shift_energy - base_energy_matlantis
                    energy_lst.append(energy)
        
        return energy_lst
        
    else:
        print(f"{energy_path} doesn't exist.")
        
        return [0]

def plot_convex_full(base_dir, cryspy_ene_lst, comp_list,
                     show_known_materials=True, matlantis_energy=[], materials_project_energy=[]):
    # Create the plot
    plt.figure(figsize=(10, 6))

    # Iterate through the data, plotting each row along the y-axis
    for i, label in enumerate(comp_list):
        y_values = cryspy_ene_lst[i]
        x_values = [label] * len(y_values)
        if i == 0:
            plt.scatter(x_values, y_values, c = "gray", label = "Generated structures by CrySPY (Matlantis)")
        else:
            plt.scatter(x_values, y_values, c = "gray")

    env_x, env_y = envelope(cryspy_ene_lst, comp_list)
    plt.plot(env_x, env_y, c = "gray", label = "Envelope for CrySPY structures")
    plt.scatter([0, 1], [0, 0])
    
    if show_known_materials:
        plt.scatter(comp_list, materials_project_energy, c = "red", label = "Known structures(Materials Project)")
        plt.scatter(comp_list, matlantis_energy, c = "blue", label = "Known structures(Matlantis)")
    
    cryspy_ene_lst_flat = list(itertools.chain.from_iterable(cryspy_ene_lst))
    combined_list = cryspy_ene_lst_flat + materials_project_energy + matlantis_energy
    min_value = min(combined_list)
    
    plt.xlim([-.1, 1.1])
    plt.ylim([min_value - 0.5, 0.5])
    plt.xlabel('Composition (fraction elem2)', fontsize = 15)
    plt.ylabel('formation energy (eV/atom)', fontsize = 15)
    plt.legend()
    plt.grid(alpha = 0.5)
    plt.savefig(base_dir + "/convex_full.png")
    
if __name__ == "__main__":
    elem1 = "Sr"
    elem2 = "P"
    MP_API_KEY = "YOUR-API-KEY"
    
    binary, ele1, ele2 = query(elem1, elem2)
    base_energy_pymatgen_e1, base_energy_matlantis_e1 = get_base_energy(ele1)
    base_energy_pymatgen_e2, base_energy_matlantis_e2 = get_base_energy(ele2)

    print(f"enegy per atom for single element(pymatgen):{base_energy_pymatgen_e1:.4f}")
    print(f"enegy per atom for single element(matlantis):{base_energy_matlantis_e1:.4f}")
    print(f"enegy per atom for single element(pymatgen):{base_energy_pymatgen_e2:.4f}")
    print(f"enegy per atom for single element(matlantis):{base_energy_matlantis_e2:.4f}")
