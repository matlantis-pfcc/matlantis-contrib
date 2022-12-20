from pfp_api_client.pfp.estimator import Estimator, EstimatorCalcMode
from pfp_api_client.pfp.calculators.ase_calculator import ASECalculator
from pfcc_extras.visualize.ngl_utils import (add_force_shape, get_struct,
                                             save_image, update_tooltip_atoms, _get_standard_pos, add_axes_shape)

import ase, os, sys, glob, traceback, datetime, pytz
from io import StringIO
import pandas as pd
import numpy as np

import autode
from ase import Atoms
from ase.constraints import FixAtoms
from ase.io import Trajectory
from ase.optimize import LBFGS
from ase.visualize.ngl import NGLDisplay
from ase.io.proteindatabank import write_proteindatabank

from ipywidgets import (widgets, Output, Layout, Text, Button, HBox, VBox, IntSlider, Label, Textarea, Select, Checkbox,
                        Box, FloatSlider)
from ipywidgets.widgets import DOMWidget
from IPython.display import HTML

import nglview as nv
from nglview import NGLWidget

from typing import Any, Dict, List, Optional, Union
from traitlets import Bunch

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

estimator = Estimator(model_version="v3.0.0", calc_mode=EstimatorCalcMode.CRYSTAL)
calculator = ASECalculator(estimator)

df = pd.DataFrame(
    columns={"SMILES": "str", "formula": "object", "atoms": "object", "energy": "float", "maxforce": "float",
             "opt": 'bool'})
pd.options.display.max_colwidth = 30


class JSMEBox:
    def __init__(self, jsme_folderpath=None):
        """JSMEBox can create SMILES from JSME drawings  .
        Parameters:
            jsme_folderpath: str  The (higher level) path where jsme.nocache.js is stored.
        """
        jsme_url = os.environ['MATLANTIS_NOTEBOOK_URL'] + "files"  # for own matlantis url
        if jsme_folderpath is None:
            jsme_folderpath = os.path.dirname(__file__)
        jsme_filepath = jsme_folderpath + "/jsme-editor.github.io/dist/jsme/jsme.nocache.js"
        if not os.path.exists(jsme_filepath):
            # Search "jsme.nocache.js" under `jsme_folderpath` directory.
            jsme_filepath_list = glob.glob(jsme_folderpath + "/**/jsme.nocache.js", recursive=True)
            print("jsme_filepath_list = ", jsme_filepath_list)
            if len(jsme_filepath_list) == 0:
                raise FileNotFoundError("jsme.nocache.js not found, please prepare JSME software.")
            jsme_filepath = jsme_filepath_list[0]
        # print("jsme_filepath = ", jsme_filepath)
        jsme_url += jsme_filepath.replace("home/jovyan", "")
        self.jsme_url = jsme_url
        jsme_matlantis_template_htm = """
        <html><head>
        <script type="text/javascript" language="javascript" src="own_jsme_url"></script>
        <script>    
        function jsmeOnLoad() {  jsmeApplet = new JSApplet.JSME("myid", "430px","285px");           
        jsmeApplet.setCallBack("AfterStructureModified", show_smiles);        }        
        function show_smiles(event) {
         smiles = event.src.smiles();            
         el = document.querySelector("input[placeholder='myid']");           
         el.value = smiles.replace("/\\[([A-Za-z][a-z]?)H?\\d*:\\d+\\]/g",smiles);           
         el.dispatchEvent(new Event('input',{'bubbles': true,'cancelable': true}));        
        }
        </script></head><body><div id="myid"></div></body></html>
        """
        nowtime = f"{datetime.datetime.now(pytz.timezone('Asia/Tokyo')):%Y%m%d%H%M%S}"
        jsme_matlantis_htm = jsme_matlantis_template_htm.replace("own_jsme_url", jsme_url).replace("myid", nowtime)
        jsmeoutput = widgets.Output(layout={"width": '440px', "height": '290px'})
        with jsmeoutput:
            display(HTML(jsme_matlantis_htm))

        smilesbox = Text(value='', placeholder=nowtime, description='SMILES',
                         layout={"width": "430px", "height": '45px'}, style={"description_width": "45px"})
        self.jsmeoutput = jsmeoutput
        self.smilesbox = smilesbox
        self.view = VBox([jsmeoutput, smilesbox])

        def __repr__(self) -> str:
            return str(display(self.view))


class Moldraw:
    """draw and get ase atoms by JSME"""

    def __init__(self, df=df, estimator=estimator):
        """Moldraw can create Ase Atoms objects and Pandas Dataframes from JSME drawings or SMILES.
                Parameters:
                 df: Pandas dataFrame object with atoms column containing Ase Atoms object.
                 estimator : pfp_api_client.pfp.estimator.Estimator
        """
        self.df = df
        self.jsme = JSMEBox()
        self.jsme.smilesbox.observe(self.smiles_change, names='value')

        button_add = Button(description='ADD IT', layout={'width': '140px'})
        button_add.on_click(self.add_atoms)
        button_conf = Button(description='Confomer serach', layout={'width': '140px'})
        button_conf.on_click(self.add_conformersdf)
        self.check_opt = Checkbox(description='OPT', indent=False, layout={'width': '80px'})

        self.dfview = Dfview(df)
        self.dfview.row_revSlider.observe(self.dfrowchange)
        self.atomslist = list(df.atoms.values)
        self.atoms = self.atomslist[0] if len(self.atomslist) > 0 else None
        self.nglbox = Nglbox(atoms=self.atoms, w=400, h=340)
        self.fmax = 0.05

        display(
            VBox([
                HBox([
                    VBox([
                        self.jsme.view,
                        HBox([self.check_opt, button_add, button_conf]),
                    ]),
                    self.nglbox.viewwithcheck
                ]),
                self.dfview.view])
        )

    def __repr__(self) -> str:
        return str(display(self.df))

    def __getitem__(self, index):
        return self.atomslist[index]

    def smiles_change(self, change):
        if change["name"] == "value" and len(self.jsme.smilesbox.value) > 0:
            atoms = smiles_to_ase(self.jsme.smilesbox.value)
            self.atoms = atoms
            self.nglbox.update_structre(atoms)

    def add_atoms(self, submit):
        smiles = self.jsme.smilesbox.value
        if len(smiles) > 0:
            atoms = self.atoms
            if atoms.calc is None:
                atoms.calc = calculator
            if self.check_opt.value:
                opt = LBFGS(atoms, maxstep=0.1, logfile=None)
                opt.run(fmax=self.fmax)

            atoms.info["SMILES"] = smiles
            e = atoms.get_potential_energy()
            maxforce = np.sqrt(((atoms.get_forces()) ** 2).sum(axis=1).max())
            formula = atoms.get_chemical_formula()
            series = pd.Series([smiles, formula, atoms.copy(), e, maxforce, self.check_opt.value],
                               index=["SMILES", "formula", "atoms", "energy", "maxforce", "opt"])
            self.df = self.df.append(series, ignore_index=True)
            self.dfview.df = self.df
            self.dfview.row_revSlider.value = - (max(0, len(self.df) - 5))
            self.dfview.showdf({"name": "value"})
            self.atomslist += [atoms]

    def add_conformersdf(self, submit):
        if len(self.jsme.smilesbox.value) > 0:
            with self.dfview.dfoutput:
                conformersdf = get_conformersdf(self.jsme.smilesbox.value, fmax=self.fmax, opt=self.check_opt.value)
            self.df = self.df.append(conformersdf, ignore_index=True)
            self.atomslist += list(self.df["atoms"].values)
            self.dfview.df = self.df
            self.dfview.row_revSlider.value = - (max(0, len(self.df)))
            self.dfview.showdf({"name": "value"})

    def dfrowchange(self, change):
        if change["name"] == "value":
            atoms = self.df.iloc[self.dfview.row].atoms
            self.nglbox.update_structre(atoms)


def smiles_to_ase(smiles):
    ele = []
    for i, sm in enumerate(smiles.split(".")):
        if len(sm) > 0:
            global ade_mol
            ade_mol = autode.Molecule(smiles=sm)
            ele += [atom.atomic_symbol for atom in ade_mol.atoms]
            pos = np.array([atom.coord for atom in ade_mol.atoms])
            if i == 0:
                posa = pos
            else:
                pos = np.array([atom.coord for atom in ade_mol.atoms]) + [max(posa[:, 0]) - min(pos[:, 0]) + 2, 0, 0]
                posa = np.vstack([posa, pos])
    return ase.Atoms(ele, posa)


def get_conformersdf(smiles, fmax=0.05, top=5, rmsd_threshold=0.2, opt=True, n_confs=100):
    conformerdf = pd.DataFrame()
    ade_mol = autode.Molecule(smiles=smiles)
    autode.config.Config.rmsd_threshold = rmsd_threshold + (ade_mol.n_atoms - 15) / 100  # 時間短縮のため原子数に応じ閾値増加
    ade_mol.populate_conformers(n_confs=n_confs)
    n = len(ade_mol.conformers)
    print(
        f"{n} conformers found (n_confs:{n_confs} n_atoms:{ade_mol.n_atoms},rmsd_threshold:{autode.config.Config.rmsd_threshold})")

    for i, ade_mol in enumerate(ade_mol.conformers):
        ele = [atom.atomic_symbol for atom in ade_mol.atoms]
        pos = np.array([atom.coord for atom in ade_mol.atoms])
        atoms = ase.Atoms(ele, pos)
        atoms.calc = calculator
        atoms.info["SMILES"] = smiles
        e = atoms.get_potential_energy()
        maxforce = np.sqrt(((atoms.get_forces()) ** 2).sum(axis=1).max())
        formula = atoms.get_chemical_formula()
        series = pd.Series([smiles, formula, atoms.copy(), e, maxforce, False],
                           index=["SMILES", "formula", "atoms", "energy", "maxforce", "opt"])
        conformerdf = conformerdf.append(series, ignore_index=True)
        conformerdf = conformerdf.astype({"opt": bool})
    conformerdf = conformerdf.sort_values("energy")
    conformerdf = conformerdf.reset_index(drop=True)
    conformerdf["conf_no"] = conformerdf.index
    if opt:
        print(f"start opt top {min(top, n)} conformers")
        for index, row in conformerdf[:top].iterrows():
            print(f" {smiles} {index + 1} / {min(top, n)} ", end="")
            atoms = row["atoms"]
            atoms.calc = calculator
            opt = LBFGS(atoms, maxstep=0.1, logfile=None)
            opt.run(fmax=fmax)
            e = atoms.get_potential_energy()
            maxforce = np.sqrt(((atoms.get_forces()) ** 2).sum(axis=1).max())
            conformerdf.loc[index, "energy"] = e
            conformerdf.loc[index, "maxforce"] = maxforce
            conformerdf.loc[index, "opt"] = True
            print("opt", opt.nsteps, "steps done")

        conformerdf = conformerdf.sort_values("energy")
        conformerdf = conformerdf.reset_index(drop=True)
        conformerdf["conf_no_opt"] = conformerdf.index
    return conformerdf


class Dfplot:
    def __init__(self, df=df, w=550, h=450):
        self.df = df
        self.w = w
        self.h = h

        plotnumbercolcands = list(df.select_dtypes(include='number').columns.values)
        plotcolcands = list(df.columns.values)
        self.plotoutput = widgets.Output(
            layout={'width': str(w) + 'px', 'height': str(h) + 'px', 'border': '1px solid gray', })
        minilayout = widgets.Layout(width='200px', height='27px', border='0px')
        ministyle = {'description_width': "30px"}
        self.w1 = Select(description='col1:', options=plotcolcands, rows=1, style=ministyle, layout=minilayout)
        self.w2 = Select(description='col2:', options=plotnumbercolcands + ["histplot col1"], rows=1, style=ministyle,
                         layout=minilayout)
        self.yycheck = Checkbox(value=False, description='yy ploy', style=ministyle, layout={'width': '100px'})
        showplotbutton = Button(description="plot", layout={'width': '100px'})
        showplotbutton.on_click(self.show_plot)
        self.df = df
        self.plotview = VBox([
            HBox([self.w1, self.w2, self.yycheck, showplotbutton]),
            self.plotoutput
        ])
        display(self.plotview)

    def show_plot(self, change):
        with self.plotoutput:
            self.plotoutput.clear_output()
            df = self.df
            col1 = self.w1.value
            col2 = self.w2.value
            fig = plt.figure(figsize=(self.w / 100, self.h / 100))
            ax = fig.add_subplot(1, 1, 1)
            if col2 == "histplot col1":
                df[col1].plot.hist(grid=True, ax=ax, bins=min(30, len(df[col1]) // 2))  # =df.target)
                plt.xlabel(col1)
            else:
                if self.yycheck.value:
                    if df[[col1, col2]].isna().sum().sum() > 0:
                        print("欠損", df[[col1, col2]].isna().sum().to_dict())
                    yydf = df[[col1, col2]].dropna()
                    yydf = yydf.loc[:, ~yydf.columns.duplicated()]
                    MAE = mean_absolute_error(yydf[col1], yydf[col2])
                    RMSE = mean_squared_error(yydf[col1], yydf[col2], squared=False)
                    R2 = r2_score(yydf[col1], yydf[col2])
                    rr = np.corrcoef(yydf[col1], yydf[col2])[0, 1]
                    plt.title(f"\n MAE:{MAE:.3f}  RMSE:{RMSE:.3f} r2_score:{R2:.3f} R:{rr:3f}")
                    YMax = yydf.max().max()
                    YMin = yydf.min().min()
                    delta01 = (YMax - YMin) * 0.1
                    YMax = YMax + delta01
                    YMin = YMin - delta01
                    plt.plot([YMin, YMax], [YMin, YMax], linestyle="dotted", lw=1)
                    plt.ylim(YMin, YMax)
                    plt.xlim(YMin, YMax)
                    mybin = np.linspace(YMin, YMax, 100)
                    H = ax.hist2d(yydf[col1].values, yydf[col2].values, bins=[mybin, mybin],
                                  norm=matplotlib.colors.LogNorm(), cmap=cm.jet)
                    ax.set_xlabel(col1)
                    ax.set_ylabel(col2)
                    plt.gca().set_aspect('equal', adjustable='box')
                    fig.colorbar(H[3], ax=ax, shrink=0.5)
                else:
                    df.plot.scatter(x=col1, y=col2, s=8, grid=True, ax=ax)
            plt.tight_layout()
            plt.show()


class Dfview():
    def __init__(self, df, w=600, h=330):
        self.row_revSlider = widgets.IntSlider(value=0, min=min(0, -(df.shape[0] - 1)), max=0, step=1, description='No',
                                               orientation='vertical')  # , readout=False )
        self.row_revSlider.observe(self.showdf)
        self.column_Slider = widgets.IntSlider(value=0, min=0, max=df.shape[1] - 1, step=1, description=str(df.shape),
                                               style={'description_width': "40px"})
        self.column_Slider.observe(self.showdf)
        self.df = df
        self.dfoutput = widgets.Output(
            layout={'width': str(w) + 'px', 'height': str(h) + 'px', 'border': '1px solid black', })
        self.row = 0
        self.view = HBox([
            self.row_revSlider,
            VBox([
                self.column_Slider,
                self.dfoutput
            ])
        ])
        self.showdf({"name": "value"})

    def __repr__(self) -> str:
        return str(display(self.view))

    def __getitem__(self, index):
        return self.atomslist[index]

    def showdf(self, change):
        if change["name"] == "value":
            df = self.df
            self.row_revSlider.min = -max(0, (df.shape[0] - 1))
            self.row = -self.row_revSlider.value
            self.row_revSlider.description = str(self.row)

            self.column_Slider.max = max(0, (df.shape[1] - 1))
            self.column_Slider.description = str(df.shape)
            self.dfoutput.clear_output()
            with self.dfoutput:
                display(self.df.iloc[
                        self.row: min(self.df.shape[0], self.row + 10),
                        self.column_Slider.value: min(self.df.shape[1], self.column_Slider.value + 7)
                        ])


class Nglbox():
    """
    """

    def __init__(self, atoms=None, w=400, h=340):
        view = nv.NGLWidget(width=str(w) + "px", height=str(h) + "px")
        self.view = view
        self.check_index = Checkbox(description='index', indent=False, layout={'width': '80px'})
        self.check_axes = Checkbox(description='axes', indent=False, layout={'width': '80px'})
        self.check_force = Checkbox(description='force', indent=False, layout={'width': '80px'})
        self.check_chage = Checkbox(description='chage', indent=False, layout={'width': '80px'})
        self.check_index.observe(self.update_deco, names="value")
        self.check_axes.observe(self.update_deco, names="value")
        self.check_force.observe(self.update_deco, names="value")
        self.check_chage.observe(self.update_deco, names="value")
        self.checkboxes = HBox([self.check_index,
                                self.check_axes,
                                self.check_force,
                                self.check_chage
                                ], layout=Layout(width='250px', height='38px'))
        self.viewwithcheck = VBox([self.view, self.checkboxes])
        self.update_structre(atoms)

    def update_deco(self, change):
        if self.atoms is None: return
        _ = [self.view.remove_component(id) for id in self.view._ngl_component_ids[1:]]
        _ = [self.view.remove_label() for i in range(3)]
        if self.check_axes.value: self.show_axes_shape()
        if self.check_force.value: self.show_force()
        if self.check_chage.value: self.show_charge_label()
        if self.check_index.value: self.show_atomindex()
        if any(self.atoms.pbc): self.view.add_unitcell()

    def update_structre(self, atoms):
        v = self.view
        self.atoms = atoms
        if atoms is None:
            return
        if len(v._ngl_component_ids) == 0:
            self.struct_component = v.add_structure(nv.ASEStructure(atoms))
        else:
            _ = [v.remove_component(id) for id in v._ngl_component_ids[1:]]
            v._remote_call('replaceStructure', target='Widget', args=get_struct(atoms))
        update_tooltip_atoms(self.view, self.atoms)
        self.update_deco("")

    def __repr__(self) -> str:
        return str(display(self.viewwithcheck))

    def show_force(self):
        force_scale: float = 3
        if self.atoms.calc is None:
            self.atoms.calc = calculator
        c = add_force_shape(self.atoms, self.view, force_scale, [1, 0, 0])

    def show_axes_shape(self):
        c = add_axes_shape(self.atoms, self.view)

    def show_charge_label(self, threshold: float = 0.05, radius: float = 1.2):
        atoms = self.atoms
        if atoms.calc is None:
            atoms.calc = calculator
        self.view.remove_label()
        charge = np.round(atoms.get_charges(), 1)
        self.view.add_label(
            color="blue",
            labelType="text",
            labelText=np.where(charge > threshold, charge, "").ravel().astype("str").tolist(),
            zOffset=2.0,
            attachment="middle_center",
            radius=radius,
        )
        self.view.add_label(
            color="rgb(255, 180, 180)",
            labelType="text",
            labelText=np.where(charge < -threshold, charge, "").ravel().astype("str").tolist(),
            zOffset=2.0,
            attachment="middle_center",
            radius=radius,
        )

    def show_atomindex(self):
        v = self.view
        atoms = self.atoms
        v.remove_label()
        v.add_label(
            color="black", labelType="text",
            labelText=[str(i) for i in range(atoms.get_global_number_of_atoms())],
            zOffset=1.0, attachment='middle_center', radius=1.5
        )
