import os
import sys
import glob
import re
from io import StringIO
import shutil
from subprocess import run, PIPE
from pathlib import Path
from io import StringIO
import shutil
from subprocess import run, PIPE
from pathlib import Path
import platform
import numpy as np
import pandas as pd

from ase import Atoms, units
from ase.io import read, write, Trajectory
from ase.calculators.emt import EMT
from ase.md.npt import NPT
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem, Draw


class EMCInterface(object):
    # Enhanced Monte Carlo\
    # https://montecarlo.sourceforge.net/emc/Welcome.html
    # https://matsci.org/c/emc/50
    #
    # P.J. in 't Veld and G.C. Rutledge, Macromolecules 2003, 36, 7358
    def __init__(
        self,
        emc_root: str = None,
        emc_version: str = "9.4.4",
        eshfilename="setup.esh",
        verbose: bool = False,
    ):
        self.settings = dict()
        self.eshfilename = eshfilename
        self.verbose = verbose
        self.emc_root = emc_root
        if self.emc_root is None:
            self.emc_root = (
                Path(__file__).parent / ".." / ".." / f"EMC/v{emc_version}"
            ).resolve()

        emc_path = [self.emc_root / "bin", self.emc_root / "scripts"]
        os.environ["PATH"] = ":".join([*[str(_) for _ in emc_path], os.environ["PATH"]])
        os.environ["EMC_ROOT"] = str(self.emc_root)
        if self.verbose:
            print(self.emc_root)
            print(emc_path)
            print(os.environ["PATH"])
            print(os.environ["EMC_ROOT"])

        system_emcexe = {
            "Darwin": "emc_macos",
            "Linux": "emc_linux_x86_64",
            "Windows": "emc_win32",
        }
        self.emc = system_emcexe[platform.system()]

    def setup(self, template, *args, **kwargs):
        if template == "homopolymer":
            self.setup_homopolymer(*args, **kwargs)
        if template == "liquid":
            self.setup_liquid(*args, **kwargs)
        if template == "mixture":
            self.setup_mixture(*args, **kwargs)

    def setup_homopolymer(
        self,
        # --> settings for emc_setup.pl
        smiles_center: str = "*c1ccc(cc1)Oc2ccc(cc2)N3C(=O)c4ccc(cc4C3=O)*",
        smiles_left: str = "Cc1ccc(cc1)Oc2ccc(cc2)N3C(=O)c4ccc(cc4C3=O)*",
        smiles_right: str = "*c1ccc(cc1)Oc2ccc(cc2)N3C(=O)c4ccc(cc4C3=O)C",
        ntotal: int = 2000,
        density: float = 0.85,
        field: str = "pcff",
        ring_depth: int = "auto",
        build_dir: str = "./build",
        lammps_prefix: str = "homopolymer",
        project: str = "homopolymer",
        #        polymer_fraction=1,
        repeat_center=8,
        repeat_left=1,
        repeat_right=1,
        seed: int = -1,
        emc_execute=False,
        #
    ):
        from esh_template_homopolymer import template

        settings = dict(
            center=smiles_center,
            left=smiles_left,
            right=smiles_right,
            field=field,
            ntotal=ntotal,
            density=density,
            ring_depth=ring_depth,
            build_dir=build_dir,
            lammps_prefix=lammps_prefix,
            project=project,
            seed=seed,
            emc_execute=emc_execute,
            #            polymer_fraction=polymer_fraction,
            repeat_center=repeat_center,
            repeat_left=repeat_left,
            repeat_right=repeat_right,
        )

        for k, v in settings.items():
            if isinstance(v, bool):
                settings[k] = "true" if v else "false"

        self.settings.update(settings)
        if self.verbose:
            print(self.settings)
        esh = template.format(**self.settings)

        with open(self.eshfilename, "w") as f:
            f.write(esh)

        cmd = f"{self.emc_root}/scripts/emc_setup.pl {self.eshfilename}".split()
        p = run(cmd, shell=False, stdout=PIPE, stderr=PIPE, text=True)
        if self.verbose:
            print(p.stdout)
            print(p.stderr)

    def setup_liquid(
        self,
        name_smiles_fractions={"water": ("O", 100)},
        # --> settings for emc_setup.pl
        ntotal: int = 2000,
        density: float = 0.85,
        field: str = "pcff",
        ring_depth: int = "auto",
        build_dir: str = "./build",
        lammps_prefix: str = "liquid",
        project: str = "liquid",
        seed: int = -1,
        emc_execute=False,
        #
    ):
        from esh_template_liquid import template

        settings = dict(
            field=field,
            ntotal=ntotal,
            density=density,
            ring_depth=ring_depth,
            build_dir=build_dir,
            lammps_prefix=lammps_prefix,
            project=project,
            seed=seed,
            emc_execute=emc_execute,
        )

        for k, v in settings.items():
            if isinstance(v, bool):
                settings[k] = "true" if v else "false"

        self.settings.update(settings)

        groups = {f"{n:15s} {s:s}" for n, (s, f) in name_smiles_fractions.items()}
        groups = "\n".join(groups)
        clusters = {
            f"{n:15s} {n:s},{f:d}" for n, (s, f) in name_smiles_fractions.items()
        }
        clusters = "\n".join(clusters)
        self.settings["groups"] = groups
        self.settings["clusters"] = clusters

        if self.verbose:
            print(self.settings)
        esh = template.format(**self.settings)

        with open(self.eshfilename, "w") as f:
            f.write(esh)

        cmd = f"{self.emc_root}/scripts/emc_setup.pl {self.eshfilename}".split()
        p = run(cmd, shell=False, stdout=PIPE, stderr=PIPE, text=True)
        if self.verbose:
            print(p.stdout)
            print(p.stderr)

    def setup_mixture(
        self,
        # --> settings for emc_setup.pl
        name_smiles_fractions={"water": ("O", 100)},
        smiles_center: str = "*c1ccc(cc1)Oc2ccc(cc2)N3C(=O)c4ccc(cc4C3=O)*",
        smiles_left: str = "Cc1ccc(cc1)Oc2ccc(cc2)N3C(=O)c4ccc(cc4C3=O)*",
        smiles_right: str = "*c1ccc(cc1)Oc2ccc(cc2)N3C(=O)c4ccc(cc4C3=O)C",
        ntotal: int = 2000,
        density: float = 0.85,
        field: str = "pcff",
        ring_depth: int = "auto",
        build_dir: str = "./build",
        lammps_prefix: str = "mixture",
        project: str = "mixture",
        polymer_fraction=1,
        repeat_center=8,
        repeat_left=1,
        repeat_right=1,
        seed: int = -1,
        emc_execute=False,
        #
    ):
        from esh_template_mixture import template

        settings = dict(
            center=smiles_center,
            left=smiles_left,
            right=smiles_right,
            field=field,
            ntotal=ntotal,
            density=density,
            ring_depth=ring_depth,
            build_dir=build_dir,
            lammps_prefix=lammps_prefix,
            project=project,
            seed=seed,
            emc_execute=emc_execute,
            polymer_fraction=polymer_fraction,
            repeat_center=repeat_center,
            repeat_left=repeat_left,
            repeat_right=repeat_right,
        )

        for k, v in settings.items():
            if isinstance(v, bool):
                settings[k] = "true" if v else "false"

        self.settings.update(settings)

        groups = [f"{n:15s} {s:s}" for n, (s, f) in name_smiles_fractions.items()]
        groups = "\n".join(groups)
        clusters = [
            f"{n:15s} {n:s},{f:d}" for n, (s, f) in name_smiles_fractions.items()
        ]
        clusters.append(f"{'poly':15s} {'alternate':s},{polymer_fraction:d}")
        clusters = "\n".join(clusters)
        self.settings["groups"] = groups
        self.settings["clusters"] = clusters

        if self.verbose:
            print(self.settings)
        esh = template.format(**self.settings)

        with open(self.eshfilename, "w") as f:
            f.write(esh)

        cmd = f"{self.emc_root}/scripts/emc_setup.pl {self.eshfilename}".split()
        p = run(cmd, shell=False, stdout=PIPE, stderr=PIPE, text=True)
        if self.verbose:
            print(p.stdout)
            print(p.stderr)

    def savefiles(self, savedir=None, emcin="build.emc", emclog="build.out"):

        if savedir is None:
            if "project" in self.settings:
                savedir = f"./build_{self.settings['project']}"
            else:
                savedir = "./build"
        os.makedirs(savedir, exist_ok=True)

        shutil.move(
            self.eshfilename, os.path.join(savedir, os.path.basename(self.eshfilename))
        )
        emcfiles = [emcin, emclog]
        for _file in glob.glob(f'./{self.settings["project"]}.*') + emcfiles:
            if not os.path.exists(_file):
                continue
            destfilename = os.path.join(savedir, os.path.basename(_file))
            shutil.move(_file, destfilename)  # overwrite

    def build(self, emcin="build.emc"):
        cmd = f"{self.emc_root}/bin/{self.emc} {emcin}".split()
        p = run(cmd, shell=False, stdout=PIPE, stderr=PIPE, text=True)
        if self.verbose:
            print(p.stdout)
            print(p.stderr)

    def clean(self):
        cmd = "emc_clean.sh"
        p = run(cmd, shell=False, stdout=PIPE, stderr=PIPE, text=True)
        return p

    def read_structure(self):
        basename = self.settings["project"]
        atoms_lammpsdata = read(f"{basename}.data", format="lammps-data")
        atoms = read(f"{basename}.pdb")
        print(len(atoms))
