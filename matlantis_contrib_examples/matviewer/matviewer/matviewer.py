import ase, ase.io ,os, sys, glob, traceback, datetime, pytz,spglib,gc,autode,threading, zipfile,scipy ,warnings ,re
from io import StringIO
import numpy as np
import pandas as pd

from ase import Atoms, units
from ase.constraints import FixAtoms,FixBondLengths,ExpCellFilter
from ase.optimize import LBFGS,FIRE
from ase.visualize.ngl import NGLDisplay
from ase.io.proteindatabank import write_proteindatabank
from ase.spacegroup.symmetrize import FixSymmetry
from ase import neighborlist
from ase.data import atomic_masses
from ase.vibrations import Vibrations
from ase.vibrations.data import VibrationsData
from ase.neb import DyNEB

from ase.md.velocitydistribution import MaxwellBoltzmannDistribution,Stationary
from ase.md.nvtberendsen import NVTBerendsen
from ase.md import MDLogger
from ase.thermochemistry import IdealGasThermo ,HarmonicThermo

from time import perf_counter

from typing import List, Optional

from ipywidgets import (widgets, Output, Layout, Text, Button, HBox, VBox, IntSlider, Label, Textarea, Select, Checkbox,
                        Box, FloatSlider, Dropdown,IntText,FloatText ,ColorPicker)
from ipywidgets.widgets import DOMWidget
from IPython.display import HTML ,Javascript

import nglview as nv
from nglview import NGLWidget
from typing import Any, Dict, List, Optional, Union
from traitlets import Bunch

from pymatgen.core.surface import SlabGenerator, generate_all_slabs
from pymatgen.symmetry.analyzer import PointGroupAnalyzer,SpacegroupAnalyzer
from pymatgen.io.ase import AseAtomsAdaptor

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from pfcc_extras.visualize.ngl_utils import (add_force_shape, get_struct,
                                             save_image, update_tooltip_atoms, _get_standard_pos, add_axes_shape)
from pfp_api_client.pfp.estimator import Estimator, EstimatorCalcMode,EstimatorMethodType
from pfp_api_client.pfp.calculators.ase_calculator import ASECalculator

pd.options.display.precision = 4
pd.options.display.max_colwidth = 30
spc =  ase.calculators.singlepoint.SinglePointDFTCalculator

class JSMEBox:
    def __init__(self,set_func= print):
        """JSMEBox can create SMILES from JSME drawings  .           
        """
        jsme_url = os.environ['MATLANTIS_NOTEBOOK_URL'] + "files"  # for own matlantis url
        jsme_folderpath = os.path.dirname(os.path.abspath(__file__))
        # __file__ はこのスクリプト自体のパス（aa.pyのパス） '' で囲わないことが重要！！！
        jsme_filepath = os.path.join(jsme_folderpath, 'data', 'jsme', 'jsme.nocache.js')
        if not os.path.exists(jsme_filepath):
            jsme_filepath_list = glob.glob(jsme_folderpath + "/**/jsme.nocache.js", recursive=True)
            if len(jsme_filepath_list) == 0:
                print(jsme_folderpath)
                print(jsme_url)
                raise FileNotFoundError("jsme.nocache.js not found, please prepare JSME software.")
            jsme_filepath = os.path.abspath( jsme_filepath_list[0] )
        
        jsme_url += jsme_filepath.replace("home/jovyan", "")
        jsme_url = re.sub(r'(?<!:)//+', '/', jsme_url) #2023/10/23 いままで上でも動いていたので互換性を持たせるためここで修正
        self.jsme_url = jsme_url
        jsme_matlantis_template_htm = """
        <html><head>
        <script type="text/javascript" language="javascript" src="own_jsme_url"></script>
        <script>    
        function jsmeOnLoad() {  jsmeApplet = new JSApplet.JSME("myid", "380px","285px");           
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
        self.jsmeoutput = widgets.Output(layout={"width": '385px', "height": '290px'})
        with self.jsmeoutput:
            display(HTML(jsme_matlantis_htm))
        self.smilesbox = Text(value='', placeholder=nowtime, description='SMILES',
                         layout={"width": "360px", "height": '45px'}, style={"description_width": "45px"})
        get_conformers_b = Button( layout ={"width":"200px"},description = "get conformers")        
        get_conformers_b.on_click(self.get_conformers_b_click)   
        self.view = VBox([self.jsmeoutput, self.smilesbox ,get_conformers_b ])
        self.smilesbox.observe(self.smiles_change, names='value')
        self.set_func = set_func
        
    def smiles_change(self,change=""):
        if change["name"] == "value" and len(self.smilesbox.value) > 0:
            atoms = smiles_to_ase(self.smilesbox.value)
            if atoms is not None:
                self.set_func( [atoms] )

    def get_conformers_b_click(self,change=""):
        latoms = get_conformers(self.smilesbox.value)
        if len(latoms) > 0:
            self.set_func( latoms )

    def __repr__(self) -> str:
        return str(display(self.view))
        
def smiles_to_ase(smiles):
    try:
        ele = []
        for i, sm in enumerate(smiles.split(".")):
            if len(sm) > 0:
                ade_mol = autode.Molecule(smiles=sm)
                ele += [atom.atomic_symbol for atom in ade_mol.atoms]
                pos = np.array([atom.coord for atom in ade_mol.atoms])
                if i == 0:
                    posa = pos
                else:
                    pos = np.array([atom.coord for atom in ade_mol.atoms]) + [max(posa[:, 0]) - min(pos[:, 0]) + 2, 0, 0]
                    posa = np.vstack([posa, pos])

        return ase.Atoms(ele, posa)
    except :
        return None

def get_conformers(smiles, rmsd_threshold=0.2, n_confs=50):
    latoms = []
    try:
        ade_mol = autode.Molecule(smiles=smiles)
        autode.config.Config.rmsd_threshold = rmsd_threshold + (ade_mol.n_atoms - 15) / 100  # 時間短縮のため原子数に応じ閾値増加
        ade_mol.populate_conformers(n_confs=n_confs)
        for ade_mol in ade_mol.conformers:
            ele = [atom.atomic_symbol for atom in ade_mol.atoms]
            pos = np.array([atom.coord for atom in ade_mol.atoms])
            latoms += [ase.Atoms(ele, pos)]
        return latoms
    except:
        if len(latoms) > 2:
            return latoms[:-1]
        return []

class Nglbox():
    """
    """
    def __init__(self, calculator=None, atoms=None, w=460, h=340):
        #view = nv.NGLWidget(width=str(w) + "px", height=str(h) + "px")
        view = nv.NGLWidget( height=str(h) + "px")
        self.view = view
        self.atoms = atoms
        self.calculator  = calculator
        self.check_index = Checkbox(description='index', indent=False, layout={'width': '80px'})
        self.check_pbc   = Checkbox(description='pbc'  , indent=False, layout={'width': '80px'})
        self.check_axes  = Checkbox(description='axes' , indent=False, layout={'width': '80px'})
        self.check_force = Checkbox(description='force', indent=False, layout={'width': '80px'})
        self.check_chage = Checkbox(description='chage', indent=False, layout={'width': '80px'})
        self.check_calc  = Checkbox(description='calc',  indent=False, layout={'width': '270px'})
        self.dEbase      = FloatText( description='base',value = 0.0 , layout={'width': '110px'} ,style={"description_width": "30px"})
        self.setebutton = Button( layout ={"width":"40px"},description = "set")        
        self.calcHbox  =  HBox( [self.check_calc] )
        self.e = 0
        self.check_calc.observe(self.show_calc, names="value")
        self.setebutton.on_click(self.setb )        
        self.check_index.observe(self.show_atomindex, names="value")
        self.check_pbc.observe  (self.show_pbc, names="value")
        self.check_axes.observe (self.show_axes_shape, names="value")
        self.check_force.observe(self.show_force, names="value")
        self.check_chage.observe(self.show_charge_label, names="value")
        self.checkboxes = HBox([self.check_index,
                                self.check_pbc,
                                self.check_axes,
                                self.check_force,
                                self.check_chage
                                ], layout=Layout(width='380px', height='38px'))
    
        frontview = Button(description="frontview", layout={'width': '140px'})
        frontview.on_click(lambda change: self.view.control.rotate(  [0, 1 / 2**0.5, 1 / 2**0.5, 0]) )
        sideview = Button(description="sideview", layout={'width': '140px'})
        sideview.on_click (lambda change: self.view.control.rotate(  [-0.5, 0.5, 0.5, 0.5]) )
        topview = Button(description="topview", layout={'width': '140px'})
        topview.on_click  (lambda change: self.view.control.rotate(  [0, 1, 0, 0]) )      
        viewsetbox = HBox([frontview ,sideview ,topview] , layout={'height': '40px'})
        self.viewwithcheck = VBox([self.view, self.checkboxes,self.calcHbox ,viewsetbox])
    
    def setb(self,change): 
        self.dEbase.value = np.round(self.e,4)
        self.show_calc()

    def update_structre(self,atoms):
        v = self.view
        self.atoms = atoms
        if atoms is None: return
        for i in range(len(v._ngl_component_ids)):
            v.remove_component(v._ngl_component_ids[0])
        v._ngl_msg_archive = []
        v._ngl_component_ids = []
        self.struct_component = v.add_structure(nv.ASEStructure(atoms))
        self.axes_shape_id  = None
        self.force_shape_id = None
        self.check_axes.value  = False
        self.check_force.value = False
        self.check_chage.value = False
        if self.check_index.value :self.show_atomindex()
        if self.check_pbc.value   :self.show_pbc()
        if self.check_calc.value  :self.show_calc()
        
    def set_struct(self,atoms):
        self.atoms = atoms
        if self.check_axes.value or self.check_force.value:
            return self.update_structre(atoms)
            
        sio = StringIO("")
        ase.io.proteindatabank.write_proteindatabank(sio, atoms)
        struct_str = sio.getvalue()
        struct = [dict(data=struct_str, ext="pdb")]
        self.view._remote_call("replaceStructure", target="Widget", args=struct)
        if self.check_calc.value  :self.show_calc()
       
    def __repr__(self) -> str:
        return str(display(self.viewwithcheck))
    
    def show_force(self,change=""):
        if self.check_force.value:
            atoms = self.atoms
            force_scale: float = 3
            if atoms.calc is None or  issubclass ( spc , type(atoms.calc) ) :
                atoms.calc = self.calculator
            self.force_shape_id = add_force_shape(atoms, self.view, force_scale, [1, 0, 0])
        else:
            if self.force_shape_id is not None:
                self.view.remove_component(self.force_shape_id)
                self.force_shape_id = None
            self.calcHbox.children = [self.check_calc,self.dEbase,self.setebutton]
    
    def show_calc(self,change=""):
        if self.check_calc.value:
            atoms = self.atoms
            if atoms.calc is None or issubclass ( spc , type(atoms.calc) )  :
                atoms.calc = self.calculator
            e = atoms.get_potential_energy()
            self.e = e
            basee = self.dEbase.value
            self.check_calc.description = f"E: {e:.4f} eV  fmax: {get_fmax(atoms):.4f}  dE: {e-basee:.3f}"
            self.calcHbox.children  = [self.check_calc, self.dEbase,self.setebutton] 
            
        else:
            self.calcHbox.children  =   [self.check_calc] 
            self.check_calc.description = "calc"
            
    def show_axes_shape(self,change=""):
        if self.check_axes.value:
            self.axes_shape_id = add_axes_shape(self.atoms, self.view)
        else:
            if self.axes_shape_id is not None:
                self.view.remove_component(self.axes_shape_id)
                self.axes_shape_id = None
                
    def show_charge_label(self,change="", threshold : float= 0.03, radius: float = 1.2 ):
        if self.check_chage.value:
            atoms = self.atoms
            if atoms.calc is None or issubclass ( spc , type(atoms.calc) ) :
                atoms.calc = self.calculator
            charge = np.round(atoms.get_charges(),1)
            self.view.add_label(
                selection = np.where(charge > threshold)[0].tolist(),
                color = "blue",
                labelType = "text",
                labelText = charge[charge > threshold].astype("str").tolist(),
                zOffset=2.0,
                attachment="middle_center",
                radius=radius,
            )
        
            self.view.add_label(
                selection = np.where(charge < -threshold)[0].tolist(),
                color="rgb(255, 180, 180)",
                labelType = "text",
                labelText = charge[charge < -threshold].astype("str").tolist(),
                zOffset=2.0,
                attachment="middle_center",
                radius=radius,
            )
            
        else:
            self.view.remove_label()
            self.view.remove_label()      

    def show_atomindex(self,change =""):
        if self.check_index.value:
            self.view.add_label(color="black", labelType="atomindex",zOffset=1.0, attachment='middle_center',radius = 1.5)
        else:
            self.view.remove_label()

    def show_pbc(self,change =""):
        if self.atoms is None:
            return
        if self.check_pbc.value:
            if sum(self.atoms.pbc) > 0:
                self.view.add_unitcell()
        else:
            self.view.remove_unitcell()

class Crystalsearch_mini:
    """search crystal from df"""
    def __init__(self ,set_func= print): 
        self.set_func= set_func
        self.df = None
        #コントロールボックスの準備
        buttonlayout = widgets.Layout(width='80px', height='30px')
        inputlayout = widgets.Layout(width='175px', height='30px')
        pyfilepath = os.path.dirname(os.path.abspath(__file__))
        self.mppath =  os.path.join(pyfilepath, 'data', 'mp.gz')
        self.search_box  = widgets.Textarea(value="Fe,O",layout=inputlayout )
        self.searchbutton = widgets.Button(description="search" ,layout=buttonlayout)
        self.searchbutton.on_click(self.search)
        self.nglbutton = widgets.Button(description="view" ,layout=buttonlayout)
        self.nglbutton.on_click(self.showngl)
        self.anyonlybox = widgets.Dropdown(description="ele option",options=["within","any","only"], layout=inputlayout)
        self.hullbox    = widgets.BoundedFloatText(value=0.3, min=0, step=0.05, description='hullabove<', layout=inputlayout ) 
        self.dfoutput  =Output(layout=Layout(overflow_y='hidden', overflow_x='hidden', height='360px' ,width = "385px"))
        
        l = Label("ex 'mp-1143','Al2O3'")
        self.contbox    = widgets.VBox([
            HBox([self.search_box,l]),
            widgets.HBox([self.anyonlybox,self.hullbox]),
            widgets.HBox([self.searchbutton,self.nglbutton]),
        ])
        self.rowslider=widgets.IntSlider(value = 0, min=0,max= 3000, step=1,description='No')
        self.rowslider.observe(self.showdf)        
        self.dfbox   = widgets.VBox( [self.rowslider,self.dfoutput] )         
        self.showcol = ["material_id","pretty_formula","energy","e_above_hull"]
        self.primitiveatomslist = []
        self.conventionalatomslist =[]
        self.view = widgets.VBox([self.contbox,self.dfbox])
    
    def search(self,change):
        if self.df is None:
            with self.dfoutput:
                print("-loading-" ,self.mppath)
                self.df = pd.read_pickle(self.mppath)
            
        self.dfoutput.clear_output()
        v = self.search_box.value
        elements = [item.strip() for item in v.split(",")]
        elements = [s for s in elements if s in list(ase.data.atomic_numbers.keys())]
        if len(elements) > 0:
            cands  = select_element(df = self.df, elements = elements, eleselect = self.anyonlybox.value )
            cands  = select_hull(cands, above_under = self.hullbox.value )
        else:
            df = self.df
            cands  = df[(df.pretty_formula == v) | (df.material_id.str.find(v) >= 0)]
            
        self.cands = cands        
        self.rowslider.max=max(len(cands)-1,1)
        self.rowslider.value = 0
        self.showdf(self)

    def showdf(self,change):
        self.dfoutput.clear_output()
        with self.dfoutput:
            display(self.cands[self.showcol].iloc[self.rowslider.value : self.rowslider.value+7])

    def showngl(self,change):
        self.primitiveatoms = self.cands.iloc[self.rowslider.value].atoms
        self.conventionalatoms = prim_to_conv(self.primitiveatoms)        
        self.set_func( [self.conventionalatoms] )
        
        
              
def select_element(df , elements = ["Fe"], eleselect = "only" ):
    """
    df: 抽出するデータフレーム    elements: 抽出する元素。リスト形式
    select: (only 、any、　within) onlyはelementsに含まれる元素のみ。anyはelementsに含まれる元素を含むレコード全部,withinは対象元素のみで構成する構造全部
    """
    if eleselect == "only":
        bool_list = [set(ele) == set(elements) for ele in df["elements"] ] # ['C', 'H']
    elif eleselect == "any":
        bool_list = [set(ele) == set(elements) | set(ele) for ele in df["elements"] ] # ['C', 'H']
    elif eleselect == "within":
        bool_list = [  np.min([ele in set(elements)  for  ele in eles]) > 0  for eles in df["elements"] ] 
    select_df = df[bool_list].reset_index(drop=True)
    return select_df

def select_hull(df, above_under = 0.05 ):
    return df[df["e_above_hull"] <= above_under ].reset_index(drop=True)
def prim_to_conv(primitiveatoms):
    lattice, scaled_positions, numbers = spglib.standardize_cell(primitiveatoms)
    conventionalatoms = ase.Atoms(numbers = numbers , cell = lattice , scaled_positions = scaled_positions ,pbc = True)
    return conventionalatoms
def get_conventional(df):
    return [ prim_to_conv(atoms) for atoms in df.atoms]    


class Mofsearch_mini:
    """search mof from df"""
    def __init__(self ,set_func= print): 
        self.set_func= set_func
        self.df = None
        #コントロールボックスの準備
        buttonlayout = widgets.Layout(width='80px', height='30px')
        inputlayout = widgets.Layout(width='175px', height='30px')
        pyfilepath = os.path.dirname(os.path.abspath(__file__))
        self.mppath =  os.path.join(pyfilepath, 'data', 'mof.gz')
        self.search_box  = widgets.Textarea(value="C, H, N ,Cu ",layout=inputlayout )
        self.searchbutton = widgets.Button(description="search" ,layout=buttonlayout)
        self.searchbutton.on_click(self.search)
        self.nglbutton = widgets.Button(description="view" ,layout=buttonlayout)
        self.nglbutton.on_click(self.showngl)
        self.anyonlybox = widgets.Dropdown(description="ele option",options=["within","any","only"], layout=inputlayout)
        self.dfoutput  =Output(layout=Layout(overflow_y='hidden', overflow_x='hidden', height='360px' ,width = "385px"))
        
        l = Label("ex 'tobacco','Cu12C84'")
        self.contbox    = widgets.VBox([
            HBox([self.search_box,l]),
            self.anyonlybox,
            widgets.HBox([self.searchbutton,self.nglbutton]),
        ])
        self.rowslider=widgets.IntSlider(value = 0, min=0,max= 3000, step=1,description='No')
        self.rowslider.observe(self.showdf)        
        self.dfbox   = widgets.VBox( [self.rowslider,self.dfoutput] )         
        self.showcol = ["qmof_id","name","formula","smiles"]
        self.primitiveatomslist = []
        self.conventionalatomslist =[]
        self.view = widgets.VBox([self.contbox,self.dfbox])
    
    def search(self,change):
        if self.df is None:
            with self.dfoutput:
                print("-loading-" ,self.mppath)
                self.df = pd.read_pickle(self.mppath)
            
        self.dfoutput.clear_output()
        v = self.search_box.value
        elements = [item.strip() for item in v.split(",")]
        elements = [s for s in elements if s in list(ase.data.atomic_numbers.keys())]
        if len(elements) > 0:
            cands  = select_element(df = self.df, elements = elements, eleselect = self.anyonlybox.value )
        else:
            df = self.df
            cands  = df[(df.qmof_id.str.find(v) >= 0) |
                        (df.name.str.find(v) >= 0) |
                        (df.formula.str.find(v) >= 0) |
                        (df.smiles.str.find(v) >= 0)
                       ]
            
        self.cands = cands        
        self.rowslider.max=max(len(cands)-1,1)
        self.rowslider.value = 0
        self.showdf(self)

    def showdf(self,change):
        self.dfoutput.clear_output()
        with self.dfoutput:
            display(self.cands[self.showcol].iloc[self.rowslider.value : self.rowslider.value+7])

    def showngl(self,change):
        self.primitiveatoms = self.cands.iloc[self.rowslider.value].atoms
        self.conventionalatoms = prim_to_conv(self.primitiveatoms)        
        self.set_func( [self.conventionalatoms] )


class File_seceltor:
    allow_file_types = ("pdb", "c3xml", "cif", "cml", "dx", "gamess", "jdx", "jxyz", "magres", "mol", "molden", "phonon", "sdf", "xodydata", "xsf", "xyz", "vasp","traj")
    def __init__(self ,set_func= None , maxlen = 500 ):      
        if set_func is None:
            setfunc = print
        self.set_func= set_func
        self.maxlen = maxlen # max files to show , max traj images  , max filesize MB
        self.file_list_box = Select(options= [""] ,layout= {"height":"400px","width":"380px"} )
        self.file_list_box.observe(self.on_file_selected , names='value')
        self.inpath_box  = Textarea(value="", placeholder='path' ,layout = {"width" :"300px"} )
        self.inpath_box   .observe(self.inpath_box_change, names='value')
        self.inpath_box.value = os.path.abspath(".") 
        up_button   = Button(description='..', layout = {"width" :"35px"} )
        h_button    = Button(description='H' , layout = {"width" :"35px"} )
        up_button.on_click  (self.up_button_click)
        h_button.on_click   (self.h_button_click)
        
        self.view = VBox ([ HBox ([self.inpath_box,up_button,h_button]) , self.file_list_box])

    def up_button_click(self,b):
        self.inpath_box.value = os.path.dirname(self.inpath_box.value)
        
    def h_button_click(self,b):
        self.inpath_box.value = os.path.abspath(".")

    def inpath_box_change(self,change):
        path = self.inpath_box.value
        file_list = []
        for t in self.allow_file_types:
            file_list += glob.glob( path + "/*." + t)
        file_list.sort(reverse=True)
        file_list = ["--- atoms file"+"-"*30 ]+ [os.path.basename(f) for f in file_list]
        if len(file_list) > self.maxlen: #max show files
            file_list = ["Displayed limit is"+str(self.maxlen ) ] + file_list[:self.maxlen]
        file_list +=["--- dirs   ---"+"-"*30]+ [ f.split("/")[-2] for f in glob.glob(path + "/*/") ]
        self.file_list_box.options = file_list

    def on_file_selected(self,change): 
        if  change['new'] is None : return
        if change['new'] != change['old'] :
            p = os.path.join(self.inpath_box.value,change['new'])
            if os.path.isdir(p):                
                self.inpath_box.value = os.path.abspath(p)
                return
            if os.path.isfile(p):
                if os.path.getsize(p) > self.maxlen * 1e6 :#500MByte
                    self.file_list_box.options = [f"File size is over {self.maxlen}MB!"] + list(self.file_list_box.options)
                    return         
                latoms = ase.io.read(p , index = ":")
                if len(latoms) > self.maxlen : # over 500 images
                    self.file_list_box.options = [f"Traj limit!{len(latoms)} summarized to  {self.maxlen}"] + list(self.file_list_box.options)
                    latoms =  latoms[::( len(latoms) // self.maxlen) + 1]
                if os.path.splitext(p)[1][1:] == "pdb": #pdb は一つ目のみセル情報が入るので2番目以降も設定
                    for i in range(len(latoms)):
                        latoms[i].pbc = latoms[0].pbc
                        latoms[i].cell = latoms[0].cell                    
                self.set_func( latoms )
                
class Cell_setter:
    def __init__(self , latoms = [Atoms("H")] , set_func= print , undo_button = Button()):
        self.latoms = latoms
        self.set_func = set_func
        
        self.get_cellinfo_button = Button( description='get_cellinfo', layout = {"width":"240px"} )
        self.out_box=Output(layout=Layout(overflow_y='scroll', overflow_x='hidden', height='220px' ,width = "385px"))
        self.cellinfo_str = Text( value = "" ,description='cellpar' ,layout = {"width":"385px"})
        self.scale_atoms_check = Checkbox(description='scale atoms' , Value = True, layout={'width': '240px'} , style={"description_width": "100px"} )
        self.set_cell_from_cellpar_button = Button( description='set_cell_from_cellpar', layout = {"width":"340px"} )
        self.scale = FloatText( min= 0.5 ,max = 2.0, value = 1.01 ,step = 0.001 , layout = {"width": "220px"} , description="scale",style={"description_width": "70px"})
        self.set_cell_scale_button = Button( description='scale cell', layout = {"width":"120px"} )
        self.view = VBox ([self.get_cellinfo_button,
                           self.out_box,
                           self.scale_atoms_check,
                           self.cellinfo_str,
                           self.set_cell_from_cellpar_button,
                           HBox([self.scale, self.set_cell_scale_button]), 
                           undo_button,
                          ])
        self.get_cellinfo_button.on_click(self.get_cellinfo_button_click)
        self.set_cell_from_cellpar_button.on_click(self.set_cell_from_cellpar_button_click)
        self.set_cell_scale_button.on_click(self.set_cell_scale_button_click)
        
    def get_cellinfo_button_click(self,b):
        with self.out_box:
            atoms = self.latoms[0]
            get_cell_info(atoms)
            if not atoms.pbc.all():    
                print("If you want to set pbc and cell, write cellpar values like 15,15,15,90,90,90 in the text box and press set_cell_from_cellpar button.")
                                            
    def set_cell_from_cellpar_button_click(self,b):
        self.out_box.clear_output()
        self.get_cellinfo_button_click("")
        atoms = self.latoms[0]
        atoms.pbc = True
        cellpar  = get_cell_from_cellparstr(cellparstr =  self.cellinfo_str.value )
        if cellpar:
            atoms    = get_newcell_atoms(atoms,cellpar ,scale_atoms=self.scale_atoms_check.value)
            self.atoms = atoms
            self.set_func( [atoms] )
            self.get_cellinfo_button_click("")

    def set_cell_scale_button_click(self,b):
        self.out_box.clear_output()
        self.get_cellinfo_button_click("")
        atoms = self.latoms[0]
        if atoms.pbc.all():
            cell = atoms.cell * self.scale.value
            atoms = get_newcell_atoms(atoms,cell ,scale_atoms=self.scale_atoms_check.value)
            self.atoms = atoms
            self.set_func( [atoms] )
            self.get_cellinfo_button_click("")
        
    def __repr__(self):
        display(self.view)
        return "cell_setter"
    
def get_cell_info(atoms):
    print("-"*30)
    print(f"{len(atoms) = }")
    if atoms.pbc.all():
        v = atoms.get_volume()
        m = sum(atoms.get_masses())
        d = m/v * 1e27 / ase.units.kg 
        print(f"a,b,c,α,γ,β = {  ','.join( map(lambda x: f'{x:.4f}', atoms.cell.cellpar())) }")    
        print(f"{d:5.4f} g/cm3  {v:5.3f} Ang3  {m:.2f} g/mol" )
    else:
        print (f"{atoms.pbc = }")
    print("-"*30)
def get_newcell_atoms(atoms,cell ,scale_atoms=True):
    atoms = atoms.copy()
    atoms.pbc = True
    atoms.set_cell( cell , scale_atoms=scale_atoms)
    return atoms
def get_cell_from_cellparstr(cellparstr =""):
    cellpar = cellparstr.split(",")
    cellpar = [ float(a.strip()) for a in cellpar if a.strip() != "" ]
    if len(cellpar) == 6:
        return cellpar
    else:
        print( cellparstr , "cant to cellpar", len(cellpar))
        return False
    
class Repeater:
    def __init__(self , latoms = [Atoms("H")] , set_func= print , undo_button = Button()):
        self.latoms = latoms
        self.set_func = set_func
        self.x  = Dropdown( options=[1,2,3,4,5], value = 2 , layout = {"width": "80px"}, description="x",style = {"description_width": "20px"})
        self.y  = Dropdown( options=[1,2,3,4,5], value = 2 , layout = {"width": "80px"}, description="y",style = {"description_width": "20px"})
        self.z  = Dropdown( options=[1,2,3,4,5], value = 1 , layout = {"width": "80px"}, description="z",style = {"description_width": "20px"})
        l_nopbc = Label("pbc allowance for no pbc atoms")
        self.allowance = FloatText( min= 0 ,max = 10, value = 3 ,step = 0.5 , layout = {"width":"140px"} , description="allowance")
        self.repeat_button = Button( description='repeat', layout = {"width":"140px"} )
        self.oder = Checkbox(description='order by z' , layout={'width': '140px'},style = {"description_width": "5px"})
        self.view = VBox ([self.x,self.y,self.z,
                           self.repeat_button ,
                           undo_button,
                           self.oder,
                           l_nopbc, self.allowance,
                          ])
        self.repeat_button.on_click(self.repeat_button_click)
        
    def repeat_button_click(self,b):
        atoms = self.latoms[0]
        atoms = atoms.copy()
        atoms.set_constraint()
        if sum(atoms.pbc) != 3:
            atoms.pbc = True
            atoms.positions -= np.min(atoms.positions , axis = 0) 
            atoms.cell = np.max( atoms.positions , axis = 0) + [self.allowance.value]*3
        atoms =  atoms.repeat([self.x.value,self.y.value,self.z.value])
        if self.oder.value:
            p1 = np.round(atoms.positions , 1)
            sorted_indices = np.lexsort((p1[:,0], p1[:,1], p1[:,2]))
            atoms =  atoms[sorted_indices]
        self.out = atoms
        self.set_func( [atoms] )
        
    def __repr__(self):
        display(self.view)
        return "repeat"
    
class Indices_setter:
    def __init__(self , latoms = None ):
        self.latoms  = latoms
        ##### atom_indices  box  #####
        self.selected_atoms_textarea = Textarea( layout = {"height":'40px',"width":'370px'} )
        self.xyz_box  = Dropdown( options=["x","y","z","i"], layout ={"width":'50px'} ,value = "z" )
        self.sign_box = Dropdown( options=["<",">="],        layout ={"width":'50px'})
        self.fs       = FloatSlider(min=0, max=100, value=5, layout ={"width":'170px'},readout_format =".1f")
        setconects    = Button(description="conects" , layout={"width": "90px"})
        setconects.on_click(self.set_connected_atoms_text)
        self.conectfrom  = IntText(value = 0 ,description="from" , layout={"width": "100px"},style = {"description_width": "35px"} )
        self.mult = FloatText(value=1.0, step = 0.05 , description = "mult", layout={"width": "100px"} ,style = {"description_width": "35px"} )
        self.view  = VBox ([self.selected_atoms_textarea , 
                            HBox ([self.xyz_box,self.sign_box,self.fs] ),
                            HBox([setconects, self.conectfrom,self.mult] ),
                           ])            
        self.fs.observe(self.set_atoms)
        
    def set_atoms(self, change):
        atoms = self.latoms[0]
        value = self.fs.value
        if self.sign_box.value == ">=":sign =  1
        if self.sign_box.value == "<" :sign = -1
        if self.xyz_box.value  == "i":
            self.fs.max = len(atoms)
            self.fs.min = 0
            self.fs.readout_format =".0f"
            self.fs.step = 1
            smols = [ i for i, atom in enumerate(atoms) if  sign* (i - value  )  >= 0 ]
        else:            
            self.fs.max = max(atoms.positions[:,self.xyz_box.index] + 0.1)
            self.fs.min = min(atoms.positions[:,self.xyz_box.index] - 0.1)
            self.fs.readout_format =".1f"
            self.fs.step = 0.1
            smols = [ i for i, atom in enumerate(atoms) if  sign*(atoms.positions[i,self.xyz_box.index] - value ) >= 0]
        self.selected_atoms_textarea.value = ", ".join(map(str, smols))
   
    def get_selected_atom_indices(self) -> List[int]:
        try:
            selected_atom_indices = self.selected_atoms_textarea.value.split(",")
            selected_atom_indices = [ int(a.strip()) for a in selected_atom_indices if a.strip() != "" ]
            return selected_atom_indices
        except Exception as e:
             selected_atoms_textarea.value  = ""
        return []
    
    def set_connected_atoms_text(self,change=[]):
        #id = self.v.picked["atom1"]["index"]
        atoms = self.latoms[0]
        atoms = atoms.copy()
        id = self.conectfrom.value
        cutOff = np.array( neighborlist.natural_cutoffs(atoms,mult = self.mult.value) )
        neighborList = neighborlist.NeighborList(cutOff, self_interaction=False, bothways=True)
        neighborList.update(atoms)
        matrix = neighborList.get_connectivity_matrix()
        n_components, component_list = scipy.sparse.csgraph.connected_components(matrix)
        connected = [ i for i in range(len(component_list)) if component_list[i] == component_list[id]]
        self.selected_atoms_textarea.value = ', '.join(map(str, connected ))        
        #self.v.observe(self._on_picked_changed_set_atoms, names=["picked"])

class Mover:
    def __init__(self , latoms = [Atoms("H")] , set_func= print , undo_button = Button()):
        self.latoms   = latoms
        self.set_func = set_func
        l1 = Label("Move Atoms indices")
        self.indices  = Indices_setter(latoms = latoms)
        l2 = Label("For Move/Rote")
        self.xyz  = Dropdown( options=["x","y","z"], description='direction' , layout = {"width":'150px'})
        self.movev = FloatSlider(value = 0 ,min = -5 , max = 5   , step = 0.1 , description='move'       , style={"description_width": "55px"}) 
        self.rotev = IntSlider( min =  0 , max = 359 , step = 1  , description='rote'  ,style = {"description_width": "55px"}) 
        move_button = Button(description='move',layout =  {"width":'140px'} )
        l3 = Label("For Set distance")
        self.disa0index  = IntText ( value = 0  ,description='fix ' ,style = {"description_width": "55px"}  ,layout =  {"width":'140px'}) 
        self.disa1index  = IntText ( value = 0  ,description='move' ,style = {"description_width": "55px"}  ,layout =  {"width":'140px'}) 
        self.distance= FloatSlider(min = 0.7 , max = 6 ,step = 0.1, description='distance' ,value = 1.6 ) 
        dis_button = Button(description='set dis',layout =  {"width":'140px'} ) 
        self.view = VBox ([
            l1,self.indices.view ,
            l2,self.xyz, self.movev,self.rotev,
            HBox ( [move_button,undo_button] ),
            l3,
            HBox ( [self.disa0index,self.disa1index] ) ,
            self.distance, 
            HBox ( [dis_button] ),
        ])
        
        move_button.on_click (self.move_button_click) 
        dis_button .on_click (self.dis_button_click)
        
    def move_button_click(self,b):
        atoms = self.latoms[0]
        atoms = atoms.copy()
        atom_indices = self.indices.get_selected_atom_indices()
        if len(atom_indices) >0 :
            temp = atoms[atom_indices]
            if self.xyz.value == "x": temp.positions +=  [self.movev.value,0,0]
            if self.xyz.value == "y": temp.positions +=  [0,self.movev.value,0]
            if self.xyz.value == "z": temp.positions +=  [0,0,self.movev.value]    
            if self.rotev.value != 0 :
                temp.rotate(self.rotev.value,self.xyz.value,center= 'COP')
            moved_pos = temp.positions
            for i,oriatomindex in enumerate(atom_indices):
                atoms[oriatomindex].position =  moved_pos[i]
            self.set_func([atoms])

    def dis_button_click(self,b):
        atoms = self.latoms[0]
        atoms = atoms.copy()
        atom_indices = self.indices.get_selected_atom_indices()
        atoms.set_distance(a0 = self.disa0index.value , 
                           a1 = self.disa1index.value , 
                           fix = 0, 
                           distance = self.distance.value ,
                           indices  = atom_indices ,
                          )
        self.set_func([atoms])
        
    def __repr__(self):
        display(self.view)
        return "Mover"

class Util_editor:
    def __init__(self , latoms = [Atoms("H")] , latoms_list =[[Atoms("H")]], set_func= print , undo_button = Button()):
        self.latoms   = latoms
        self.latoms_list = latoms_list 
        self.set_func = set_func
        l1 = Label("Edit Atoms indices")
        self.indices  = Indices_setter(latoms = latoms)
        self.delete  = Button(description="Delete")
        self.replace = Button(description="Replace to")
        self.replace_symbol = Text(value="H", layout={"width": "50px"})
        self.delete.on_click(self.delete_on_click)
        self.replace.on_click(self.replace_on_click)
        l2 = Label("Sort all Atoms")
        sortz_b  =Button(description="by pos z")
        self.sortallow = FloatText(value =0.1, step = 0.05,description="pos_allow")
        sortsym_b = Button(description="by symbols")
        sortz_b.on_click(self.sortz_on_click)
        sortsym_b.on_click(self.sortsym_on_click)
        l3 = Label("wrap Atoms")
        wrap_b  =Button(description="wrap this")
        wrap_traj_b = Button(description="wrap traj")
        self.wrapindex = IntText( description='#' ,layout={"width":"90px"} , style={"description_width": "50px"} ) 
        wrap_b.on_click(self.wrap_on_click)
        wrap_traj_b.on_click(self.wrap_traj_on_click)
                           
        self.view = VBox ([
            l1,self.indices.view ,
            self.delete,
            HBox([self.replace, self.replace_symbol]) ,
            undo_button,
            l2,
            HBox([sortz_b,self.sortallow]),
            sortsym_b,
            l3,
            wrap_b,
            HBox([wrap_traj_b,self.wrapindex]),
        ])
                           
    def delete_on_click(self, clicked_button: Button):
        atoms = self.latoms[0]
        atoms = atoms.copy()
        atom_indices = self.indices.get_selected_atom_indices()
        del atoms[atom_indices]
        self.set_func([atoms])
        
    def replace_on_click(self, clicked_button: Button):
        atoms = self.latoms[0]
        atoms = atoms.copy()
        atom_indices = self.indices.get_selected_atom_indices()
        replace_symbol = self.replace_symbol.value
        try:
            atoms.symbols[atom_indices] = replace_symbol
        except:
            return
        self.set_func([atoms])
                           
    def sortz_on_click(self,b):
        atoms = self.latoms[0]            
        p1 = np.round(atoms.positions * (0.1 / max(self.sortallow.value,1e-20) ), 1)
        sorted_indices = np.lexsort((p1[:,0], p1[:,1], p1[:,2]))
        atoms =  atoms[sorted_indices]
        self.set_func([atoms])
    
    def sortsym_on_click(self,b):
        atoms = self.latoms[0]
        atoms = atoms[ np.argsort(atoms.positions[:, 2])]
        symbols = atoms.get_chemical_symbols()
        cids = [i for i, symbol in enumerate(symbols) if symbol =="C"]
        hids = [i for i, symbol in enumerate(symbols) if symbol =="H"]
        otherids = [i for i in range(len(symbols)) if i not in cids+hids ]
        atoms = atoms[cids + hids + otherids ]
        self.set_func([atoms])
        
    def sortz_on_click(self,b):
        atoms = self.latoms[0]            
        p1 = np.round(atoms.positions * (0.1 / max(self.sortallow.value,1e-20) ), 1)
        sorted_indices = np.lexsort((p1[:,0], p1[:,1], p1[:,2]))
        atoms =  atoms[sorted_indices]
        self.set_func([atoms])
                           
    def wrap_on_click(self,b):                      
        atoms = self.latoms[0]            
        atoms.wrap()
        self.set_func([atoms])
    def wrap_traj_on_click(self,b):
        latoms = self.latoms_list[self.wrapindex.value]
        for i in range(len(latoms)):
            latoms[i].wrap()
        self.set_func(latoms)
         
    def __repr__(self):
        display(self.view)
        return "Util_editor"
    
class Surface_maker:
    def __init__(self , latoms = None , set_func= None , undo_button = None):
        self.latoms  = latoms
        self.set_func = set_func
        self.x = Dropdown( options=[0,1,2,3,4,5], value = 1 , layout = {"width": "140px"}, description="millerx")
        self.y = Dropdown( options=[0,1,2,3,4,5], value = 1 , layout = {"width": "140px"}, description="millery")
        self.z = Dropdown( options=[0,1,2,3,4,5], value = 1 , layout = {"width": "140px"}, description="millerz")
        self.min_slab_size = IntText(description='min_slab'   ,value=15 ,  layout={"width": "140px"} )
        self.min_vacuum    = IntText(description='min_vacuum' ,value=30,  layout={"width": "140px"} ) 
        sur_button    = Button(description='surface_done',layout = {"width": "140px"})
        sur_button.on_click( self.sur_button_click )
        self.view = VBox ([ self.x , self.y , self.z , self.min_slab_size ,  self.min_vacuum , sur_button  ,undo_button] )
        
    def sur_button_click(self,b):
        atoms = self.latoms[0]
        if  sum(atoms.pbc) == 3:
            atoms.set_constraint()
            bulk_structure = ase_to_pymatgen(atoms)
            slabgen = SlabGenerator(
                bulk_structure, 
                miller_index   =(self.x.value,self.y.value,self.z.value),
                min_slab_size  = self.min_slab_size.value,
                min_vacuum_size= self.min_vacuum.value ,
                primitive = False ,
                lll_reduce = True
            )
            slabs = slabgen.get_slabs(symmetrize=False)
            atoms = pymatgen_to_ase(slabs[0])
            self.set_func( [atoms] ,addnewno = True )
            
    def __repr__(self):
        display(self.view)
        return "Surface_maker"
    
def ase_to_pymatgen(atoms):
    return AseAtomsAdaptor.get_structure(atoms)
def pymatgen_to_ase(structure):
    return AseAtomsAdaptor.get_atoms(structure)

class Attach_maker:
    def __init__(self , latoms_list = None , set_func= None , undo_button = None, calculator = None):
        self.latoms_list = latoms_list 
        self.set_func = set_func
        self.calculator = calculator
        l = Label("Attach Atoms A and B with distance atom A[a] and B[b].")
        self.An     = IntText( description='A#'  ,layout={"width" : "170px"} , style={"description_width": "80px"} ) 
        self.Ann    = IntText( description='##'  ,layout={"width" : "150px"} , style={"description_width": "30px"} ) 
        self.aindex = IntText ( description='a'  ,layout={"width" : "180px"} )     
        self.Bn     = IntText( description='B#'  ,layout={"width" : "170px"} , style={"description_width": "80px"} )
        self.Bnn    = IntText( description='##'     ,layout={"width" : "150px"} , style={"description_width": "30px"} )
        self.bindex = IntText ( description='b' ,layout={"width" : "180px"} ) 
        self.distance= FloatSlider(min = 0.7 , max = 6 ,step = 0.1, description='distance'  ,value = 1.6 ) 
        self.withopt=  IntSlider(min = 0 , max = 100 , description='Opt_n'  ,value = 0)
        self.outbox = Output( layout =  {"width" :'350px' ,"height" : '220px'}  )
        attach_button = Button(description='Attach',layout = {"width" : "80px"}) 
        attach_button.on_click(self.attach_button_click)
        attach_button2 = Button(description='Just A+B',layout = {"width" : "80px"}) 
        attach_button2.on_click(self.attach_button_click2)
        self.view  = VBox ( [l,
                             HBox([self.An,self.Ann]),
                             self.aindex,
                             HBox([self.Bn,self.Bnn]),                        
                             self.bindex,
                             self.distance,
                             self.withopt, 
                             attach_button,
                             attach_button2,
                             undo_button,self.outbox ])

    def attach_button_click(self,b):
        latoms_list = self.latoms_list
        self.outbox.clear_output()
        with self.outbox:
            atoms0 = latoms_list[self.An.value][self.Ann.value].copy()
            atoms1 = latoms_list[self.Bn.value][self.Bnn.value].copy()
            atoms  = get_attach_atoms(atoms0,atoms1,self.aindex.value,self.bindex.value,
                                      distance = self.distance.value ,
                                      optsteps = self.withopt.value ,
                                      calculator = self.calculator)
            self.set_func( [atoms] )

    
    def attach_button_click2(self,b):
        latoms_list = self.latoms_list
        self.outbox.clear_output()
        with self.outbox:
            atoms0 = latoms_list[self.An.value][self.Ann.value].copy()
            atoms1 = latoms_list[self.Bn.value][self.Bnn.value].copy()
            atoms  = atoms0 + atoms1
            atoms.cell = atoms0.cell
            atoms.pbc  = atoms0.pbc
            self.set_func( [atoms] )
            
    def __repr__(self):
        display(self.view)
        return "Attach_maker"
    
def ballv(N=1000):
    '''
    Return N(x,y,z)positions which surface coordinates of a sphere of radius 1.
    '''
    f = (np.sqrt(5)-1)/2
    arr = np.linspace(-N, N, N*2+1)
    theta = np.arcsin(arr/N)
    phi = 2*np.pi*arr*f
    x = np.cos(theta)*np.cos(phi) 
    y = np.cos(theta)*np.sin(phi) 
    z = np.sin(theta) 
    return np.array([x,y,z]).T
ballvec = ballv()

def get_far_position(atoms,index,distance = 3.0):
    '''
    Return The position where the distance from the specified atom is set 
    and the longest from other atoms
    '''
    positions = atoms.positions
    cands = ballvec * distance +  positions[index]
    nears_ind = np.argsort ( atoms.get_distances(index, list(range(len(atoms))) ,mic =True) )[:50]
    checkatoms = atoms[nears_ind]
    mindis = []
    for pos in cands: 
        checkatoms[0].position = pos
        mindis += [min(checkatoms.get_distances(0, list(range(1,len(checkatoms) )) ,mic =True))]
    mindis = np.array(mindis)
    return cands[np.argmax(mindis)]#最短が最長となる座標

def get_min_distance_ofaddatoms(atoms,reactindex0,reactindex1,lenoriginalatoms):
    '''
    Return  the nearest distance between the added atoms and the non-reacting atom of the original atom
    atoms : original atoms + added atoms
    reactindex0 : int  the reacting atom index of original atoms
    lenoriginalatoms : int  original atoms number 
    ''' 
    baseatoms = atoms[:lenoriginalatoms]
    nears_ind = np.argsort ( baseatoms.get_distances(reactindex0, list(range(lenoriginalatoms)) ,mic =True) )[1:100]
    baseatoms= baseatoms[nears_ind]
    
    reactatoms = atoms[lenoriginalatoms:]
    nears_ind2 = np.argsort ( reactatoms.get_distances(reactindex1-len(atoms), list(range(len(atoms)-lenoriginalatoms)) ,mic =True) )[:100]
    reactatoms =  reactatoms[nears_ind2]
    mindis = []
    for pos in baseatoms.positions:
        reactatoms[0].position = pos
        mindis += [min(reactatoms.get_distances(0, list(range(1,len(reactatoms) )) ,mic =True))]
                                 
    return min(mindis)

def get_attach_atoms(atoms0,atoms1,a0i,a1i,distance = 1.6 ,optsteps = 0 ,calculator= None ):
    '''
    Returns an ATOMS object with a structure where the distance 
    between atoms0[a0i] and atoms1[a1i] is "distance" 
    and the distance other than the above two atoms is the maximum.
    '''
    atoms0_cp = get_far_position(atoms0,a0i,distance )
    atoms1_cp = get_far_position(atoms1,a1i,distance )
    vec0 = atoms0_cp  - atoms0.positions[a0i] + 0.000001
    vec1 = atoms1_cp  - atoms1.positions[a1i] - 0.000001
    atoms1.positions += atoms0_cp - atoms1.positions[a1i]
    atoms1.rotate(-vec1 , vec0 ,center=atoms1.positions[a1i] )
    gmin = []
    roteatoms = atoms1.copy()
    for i in range(36):
        roteatoms.rotate( 10 ,vec0 ,center=roteatoms.positions[a1i])
        tempatoms = atoms0+roteatoms
        gmin += [get_min_distance_ofaddatoms(tempatoms,a0i,a1i+len(atoms0) , len(atoms0) )]
    mid = np.argmax(gmin)   
    atoms1.rotate( mid * 10 ,vec0 ,center=atoms1.positions[a1i])
    atoms =  atoms0 + atoms1
    atoms.calc = calculator
    if optsteps > 0:
        c = FixBondLengths([[a0i, a1i+len(atoms0)]])
        atoms.set_constraint(c)
        print("opt constraint Fixbondlength")
        atoms = myopt(atoms,steps=optsteps)
        atoms.set_constraint()

    return atoms



class Traj_setter:
    def __init__(self , latoms_list = None , reset_selector_view_func= None , undo_button = None, calculator = None):
        self.latoms_list = latoms_list 
        self.reset_selector_view_func = reset_selector_view_func
        self.calculator = calculator
        l = Label("To change traj (latoms_list)")
        self.An     = IntText( description='A#'  ,layout={"width" : "170px"} , style={"description_width": "80px"} ) 
        self.Ann    = IntText( description='##'  ,layout={"width" : "150px"} , style={"description_width": "30px"} ) 
        
        self.Bn     = IntText( description='B#'  ,layout={"width" : "170px"} , style={"description_width": "80px"} )
        self.Bnn    = IntText( description='##'     ,layout={"width" : "150px"} , style={"description_width": "30px"} )
    
        self.withopt=  IntSlider(min = 0 , max = 100 , description='Opt_n'  ,value = 0)
        self.outbox = Output( layout =  {"width" :'350px'}  )
        insertBnn_afterAnn_button = Button(description='Insert B[##] after A[##]',layout = {"width" : "380px"}) 
        insertBnn_afterAnn_button.on_click(self.insertBnn_afterAnn_button_click)
        insertallB_afterAnn_button = Button(description='Insert allB after A[##]',layout = {"width" : "380px"}) 
        insertallB_afterAnn_button.on_click(self.insertallB_afterAnn_button_click)
        swap_button = Button(description='Swap A[##] and B[##]',layout = {"width" : "380px"}) 
        swap_button .on_click(self.swap_button_click)
        
        self.view  = VBox ( [l,
                             HBox([self.An,self.Ann]),
                             HBox([self.Bn,self.Bnn]),                        
                             insertBnn_afterAnn_button,
                             insertallB_afterAnn_button,
                             swap_button ,
                             self.outbox ])

    def insertBnn_afterAnn_button_click(self,b):
        latoms_list = self.latoms_list
        atomsB = latoms_list[self.Bn.value][self.Bnn.value].copy()
        with self.outbox:
            insertid = self.Ann.value + 1
            if self.Ann.value == -1:
                insertid = len(latoms_list[ self.An.value])
                
            latoms_list[self.An.value].insert( insertid, atomsB)
            self.reset_selector_view_func()
    
    def insertallB_afterAnn_button_click(self,b):
        latoms_list = self.latoms_list
        latomsB = latoms_list[self.Bn.value].copy()
        latomsA = latoms_list[self.An.value].copy()
        with self.outbox:
            insertid = self.Ann.value + 1
            if self.Ann.value == -1:
                insertid = len(latoms_list[ self.An.value])
            
            latoms_list[self.An.value] = latomsA[:insertid ] + latomsB + latomsA[insertid:] 
            self.reset_selector_view_func()

    def swap_button_click(self,b):
        latoms_list = self.latoms_list
        atomsA = latoms_list[self.An.value][self.Ann.value].copy()
        atomsB = latoms_list[self.Bn.value][self.Bnn.value].copy()
        with self.outbox:
            latoms_list[self.An.value][self.Ann.value] = atomsB
            latoms_list[self.Bn.value][self.Bnn.value] = atomsA
            self.reset_selector_view_func()
            
    def __repr__(self):
        display(self.view)
        return "Traj_setter"
    
class Neb_maker:
    def __init__(self , latoms_list = None , set_func= None , undo_button = None, calculator = None):
        self.latoms_list = latoms_list 
        self.set_func = set_func
        self.calculator = calculator
        self.initialstate0 = IntText( description='ini #' ,layout={"width" : "130px"} , style={"description_width": "50px"} ) 
        self.initialstate1 = IntText( description='##'    ,layout={"width" : "130px"} , style={"description_width": "30px"} ) 
        self.finalstate0   = IntText( description='fin #' ,layout={"width" : "130px"} , style={"description_width": "50px"} )
        self.finalstate1   = IntText( description='##' ,   layout={"width" : "130px"} , style={"description_width": "30px"} )
        self.nimages      = IntText( description='nimages' ,value = 15   ,step = 5    ,layout={"width" : "125px"} , style={"description_width": "50px"} ) 
        self.check_climb  = Checkbox ( description='climb' ,value = True ,             layout={"width" : "125px"} , style={"description_width": "50px"}) 
        self.neb_k        = FloatText( description='neb k' ,value = 0.1  ,step = 0.05 ,layout={"width" : "125px"} , style={"description_width": "50px"}) 
        self.neb_steps    = IntText(  description='steps'  ,value = 50   ,step = 5    ,layout={"width" : "125px"} , style={"description_width": "50px"})
        self.neb_fmax     = FloatText( description='fmax'  ,value = 0.1  ,step = 0.01 ,layout={"width" : "125px"} , style={"description_width": "50px"}) 
        self.run_neb_button = Button(description='run NEB',layout = {"width" : "150px"}) 
        self.run_neb_button.on_click(self.run_neb_button_click)  
        self.re_neb_images= IntText( description='reNEB images',layout={"width" : "150px"} ) 
        self.run_re_neb_button = Button(description='rerun',layout =    {"width" : "80px"}) 
        self.run_re_neb_button.on_click(self.run_re_neb_button_click)
        self.out_box=Output(layout=Layout(overflow_y='scroll', overflow_x='hidden', height='320px' ,width = "385px"))
        self.plotb  =Button(description='plot',layout = {"width" : "150px"}) 
        self.plotb.on_click(self.plotb_click)     
        self.view  = VBox ( [
            HBox([self.initialstate0 ,self.initialstate1]),
            HBox([self.finalstate0   ,self.finalstate1]),                            
            HBox([self.nimages ,self.check_climb]),
            HBox([self.neb_k , self.neb_fmax , self.neb_steps]),                           
            self.run_neb_button,              
            HBox([self.re_neb_images, self.run_re_neb_button]),
            self.out_box
        ])
 
    def run_neb_button_click(self,b):
        latoms_list = self.latoms_list
        atoms0 = latoms_list[self.initialstate0.value][self.initialstate1.value].copy()
        atoms1 = latoms_list[self.finalstate0.value][self.finalstate1.value].copy()    
        self.images =  get_interpolate_image(atoms0, atoms1,nimages = self.nimages.value)
        self.run_neb()
              
    def run_re_neb_button_click(self,b):
        latoms_list = self.latoms_list
        self.images = latoms_list[self.re_neb_images.value].copy()
        self.run_neb()
    
    def run_neb(self):
        self.out_box.clear_output()
        with self.out_box:
            images = self.images
            print( images[0].get_chemical_formula() , "nimages" ,len(images))
            if len(images)< 3 :
                print("images < 3 check re neb images index")
                return
            if images[0].get_chemical_formula() !=  images[-1].get_chemical_formula():
                print("images dont have same formura,check re neb images index.",
                      images[-1].get_chemical_formula()) 
                return
            
            name = images[0].get_chemical_formula() + "_" + nowtime() 
            trajfile = "./output/neb/"    + name + ".traj" 
            logfile = "./output/neb/log/" + name + ".csv" 
            images = neb_calc(
                images,
                climb      = self.check_climb.value , 
                k          = self.neb_k.value, 
                steps      = self.neb_steps.value , 
                fmax       = self.neb_fmax.value  ,
                calculator = self.calculator ,
                logfile    = logfile
            )
            ase.io.write(trajfile ,images)
            self.set_func( images ,addnewno = True) 
            display(self.plotb)
            self.logfile =  logfile
    
    def plotb_click(self,b):
        logfile = self.logfile
        clear_output_wo_edges(self.out_box, head=4, tail=1)
        with self.out_box:
            fig = neb_plot(logfile)
            fig.show()
            
    def __repr__(self):
        display(self.view)
        return "Neb_maker"
    
def get_interpolate_image(start, end, nimages = 15):
    configs = [start.copy() for i in range(nimages-1)] + [end.copy()] 
    neb = DyNEB(configs)
    neb.interpolate(mic=True)
    return configs

def neb_calc(configs,  climb = True, k = 0.1, steps=200, fmax = 0.05,calculator=None, logfile =None ):
    model_version = calculator.estimator.model_version
    calc_mode     = calculator.estimator.calc_mode
    for atoms in configs:
        atoms.calc =  ASECalculator(Estimator(model_version=model_version, calc_mode=calc_mode))
        atoms.get_forces()
    if climb:
        neb   = DyNEB(configs, k=k, fmax = fmax, climb=True, dynamic_relaxation=True, scale_fmax=0.2)#Climbing image
        relax = FIRE (neb, trajectory=None)
    else:    
        neb = DyNEB(configs, k=k, fmax = fmax, climb=True, dynamic_relaxation=True, scale_fmax=0.2)#Climbing image
        relax = LBFGS(neb, trajectory=None)
    if logfile is not None:
        ise = configs[0 ].get_potential_energy()
        fse = configs[-1].get_potential_energy()
        def neb_logging():
            l = [relax.nsteps] + [ise] + list(neb.energies[1:-1]) + [fse] +["\n"]
            with open(logfile, 'a') as f:
                f.write(" ".join(map(str, l)) )
        relax.attach(neb_logging , interval=1 )    
         
    relax.run(fmax=fmax, steps=steps)#Fmax は　NEB全ビーズ構造の最大フォース値    
    return configs   

def neb_plot(neblogfile,headrows=5 , maxrows = 9 ,h = 260 ,w = 360 ):
    df = pd.read_csv(neblogfile , header=None,delim_whitespace=True,)
    df.set_index(0, inplace=True)
    df.index.name = None
    df.columns = range(len(df.columns))
    fig = go.Figure()
    for idx, row in df.head(headrows).iterrows():
        fig.add_trace(go.Scatter(x=row.index, y=row.values, name=str(idx)))
    for i in range( headrows , len(df)-1 , max(1,  (len(df) - headrows ) // (maxrows-headrows) ) ):
        fig.add_trace(go.Scatter(x=df.columns, y=df.iloc[i], name=str(i)))
    fig.add_trace(go.Scatter(x=df.columns, y=df.iloc[-1], name=str(len(df)-1))) 
    fig.update_layout(height=h, width=w,margin=dict(l=5, r=5, t=5, b=5))
    return fig
            
class NVT_maker:
    def __init__(self , latoms = None , set_func= None  ,calculator = None):
        self.latoms  = latoms
        self.set_func = set_func
        self.calculator = calculator 
        l = Label("NVT const calc　")
        self.time_step_box    = FloatText(description='time_step fs',value=1    , step=0.5, layout={"width" : "180px"} )
        self.temperature_box  = FloatText(description='temperature' ,value=473  , step=10 , layout={"width" : "180px"} )
        self.num_md_steps_box = IntText(description=  'md_steps'    ,value=1000 , step=100, layout={"width" : "180px"} )
        self.num_interval     = IntText(description=  'interval'    ,value =50  , step=50 , layout={"width" : "180px"} )
        self.taut_box         = FloatText(description='taut fs'     ,value=1    , step=0.5, layout={"width" : "180px"} )
        self.output_filename  = Text(description='path',value="./output/md/NVT_" ,          layout={"width" : "380px"} )
        run_button = Button(description='NVT run',layout ={"width" : "140px"} ) 
        run_button.on_click(self.run_button_click)
        self.out_box=Output(layout=Layout(overflow_y='scroll', overflow_x='hidden', height='220px' ,width = "385px"))
        self.view = VBox ([
            l,
            self.time_step_box,
            self.temperature_box,
            self.num_md_steps_box ,
            self.num_interval,
            self.taut_box,
            self.output_filename,
            run_button,
            self.out_box,
        ])

    def run_button_click(self,b):
        atoms = self.latoms[0]
        if not atoms.pbc.all():
            print("set pbc before run.")
            return
        if atoms.calc is None or issubclass ( spc , type(atoms.calc) ) :  
            atoms.calc = self.calculator
        fname = atoms.get_chemical_formula() + "_" + nowtime()
        self.out_box.clear_output()
        with self.out_box:             
            log_filename = self.output_filename.value + fname + ".log"
            print("log: ",log_filename)
            traj_filename = self.output_filename.value+ fname + ".traj"
            print("traj: ",traj_filename)
            
            time_step= self.time_step_box.value
            temperature = self.temperature_box.value
            num_md_steps = self.num_md_steps_box.value
            num_interval = self.num_interval.value
            taut = self.taut_box.value
            MaxwellBoltzmannDistribution(atoms, temperature_K=temperature,force_temp=True)
            Stationary(atoms)  # Set zero total momentum to avoid drifting
            # run MD
            dyn = NVTBerendsen(atoms, time_step*units.fs, temperature_K = temperature, taut=taut*units.fs, loginterval=num_interval, trajectory=traj_filename)
            # Print statements
            def print_dyn():
                imd = dyn.get_number_of_steps()
                epot  = atoms.get_potential_energy()
                etot  = atoms.get_total_energy()
                temp_K = atoms.get_temperature()
                stress = atoms.get_stress(include_ideal_gas=True)/units.GPa
                stress_ave = (stress[0]+stress[1]+stress[2])/3.0 
                elapsed_time = perf_counter() - start_time
                print(f"{imd: >3}  {epot:.3f}  {etot:.3f}  {temp_K:.1f}  {elapsed_time:.1f}") 
                #{stress_ave:.2f} {stress[0]:.2f} {stress[1]:.2f} {stress[2]:.2f} {stress[3]:.2f} {stress[4]:.2f} {stress[5]:.2f}
            dyn.attach(print_dyn, interval=num_interval)
            dyn.attach(MDLogger(dyn, atoms, log_filename, header=True, stress=True, peratom=True, mode="a"), interval=num_interval)
            # Now run the dynamics
            start_time = perf_counter()
            print(f"imd  Epot(eV)   Etot(eV)  T(K) elapsed_time(sec)")
            dyn.run(num_md_steps)
            images= ase.io.read(traj_filename,index=":")
            self.set_func( images )
            
    def __repr__(self):
        display(self.view)
        return "NVT_maker"

class Vib_maker:
    def __init__(self , latoms = None , set_func= None  ,calculator = None):
        self.latoms  = latoms
        self.set_func = set_func
        self.calculator = calculator 
        self.delta_box    = FloatText(description='delta',value=0.1 ,  layout={"width" : "180px"} )
        l = Label("VIB Atoms indices.If None,all atoms are targeted.")
        self.indices  = Indices_setter(latoms = latoms)
        vib_button = Button(description='vib run',layout ={"width" : "140px"} ) 
        vib_button.on_click(self.vib_button_click)
        vib_readbutton = Button(description='vib read',layout ={"width" : "140px"} ) 
        vib_readbutton.on_click(self.vib_readbutton_click)
        self.vib_mode= Dropdown( options= range(60) ,description='mode',layout={"width" : "180px"} )        
        remode_button = Button(description='make mode traj',layout ={"width":"250px"})
        remode_button.on_click(self.remode_button_click)
        self.temperature = FloatText(value=297.15,description='tempK',layout ={"width":"150px"}, style={"description_width": "55px"} )
        self.out_box=Output(layout=Layout(overflow_y='scroll', overflow_x='hidden', height='200px' ,width = "385px"))
        self.view = VBox ([
            self.delta_box,
            l,
            self.indices.view,
            HBox([vib_button,vib_readbutton]),
            self.out_box,
            HBox( [self.vib_mode, self.temperature]),
            remode_button
        ])
        
    def vib_button_click(self,b):
        atoms = self.latoms[0]
        if atoms.calc is None or issubclass ( spc , type(atoms.calc) ) :  
            atoms.calc = self.calculator
        self.out_box.clear_output()
        with self.out_box: 
            indices = self.indices.get_selected_atom_indices()
            if len(indices) == 0:
                indices= list(range(len(atoms)))
            if len(indices) > 500:
                print(len(indices) ," Too many indices.Must under 500."   )
                return 
            self.vib_mode.options = range(len(indices)*3)
            delta = self.delta_box.value
            H = my_2d_hess(atoms,delta=delta, indices=indices,thread_n =12,calculator =self.calculator)
            self.vib =   Vibrations(atoms, delta=delta, indices=indices, nfree=2)
            self.vdata = VibrationsData.from_2d(atoms, H,indices=indices)
            self.vib.H = H
            self.vib._vibrations = self.vdata
            self.vib.summary()
            atoms.info["vib_H"] = H
            atoms.info["vib_delta"] = delta
            atoms.info["vib_indices"] = None if len(indices) == len(atoms) else indices
            atoms.info["epot"] = atoms.get_potential_energy()
            name = atoms.get_chemical_formula() + "_" +nowtime()
            self.vibfile = "./output/vib/" + name + ".traj"
            atoms.write(self.vibfile)
            self.atoms = atoms
            self.set_func( [atoms] )

    def vib_readbutton_click(self,b):
        atoms = self.latoms[0]
        self.out_box.clear_output()
        with self.out_box: 
            delta   = atoms.info.get("vib_delta",0)
            if delta == 0:
                print("Atoms has no vibrational info.")
                return    
            H       = atoms.info.get("vib_H")             
            indices = atoms.info.get("vib_indices")
            epot    = atoms.info.get("epot")
            self.delta_box.value = delta
            if indices is not None:
                self.indices.selected_atoms_textarea.value = ", ".join(map(str,indices ))
            self.vib =   Vibrations(atoms, delta=delta, indices=indices, nfree=2)
            self.vdata = VibrationsData.from_2d(atoms, H,indices=indices)
            self.vib.H = H
            self.vib._vibrations = self.vdata
            self.vib.summary()
            images = [i for i in self.vdata.iter_animated_mode(0)]
            self.atoms = atoms
            images[0] = atoms
            self.set_func( images )
            
    def remode_button_click(self,b):
        images = [i for i in self.vdata.iter_animated_mode(
            self.vib_mode.value,
            temperature = units.kB * self.temperature.value,
        ) ]
        images[0] = self.atoms
        self.set_func( images )
    
    def __repr__(self):
        display(self.view)
        return "Vib_maker"
    
class Thermo_maker:
    def __init__(self , latoms = None , set_func= None  ,calculator = None):
        self.latoms  = latoms
        self.set_func = set_func
        self.calculator = calculator 
        self.temp_box   = FloatText(description ='temp_k',value=298   , layout={"width" : "160px"} )
        self.press_box  = FloatText(description='P_kPa ' ,value=101.3 , layout={"width" : "160px"} )
        ideal_button = Button(description='Idealgas ',layout ={"width" : "80px"} ) 
        ideal_button.on_click(self.ideal_button_click)
        harmo_button = Button(description='Harmonic',layout ={"width" : "80px"} ) 
        harmo_button.on_click(self.harmo_button_click)
        self.out_box=Output(layout=Layout(overflow_y='scroll', overflow_x='hidden', height='380px' ,width = "385px"))
        self.view = VBox ([
            HBox ([ideal_button , harmo_button]),
            HBox( [self.temp_box,self.press_box ]),
            self.out_box
        ])
        
    def set_vib(self,atoms):
        delta   = atoms.info.get("vib_delta",0)
        if delta == 0:
            print("Atoms has no vibrational info.")
            return False
        H       = atoms.info.get("vib_H")             
        self.indices = atoms.info.get("vib_indices",None)
        self.epot    = atoms.info.get("epot",0)
        self.vib =   Vibrations(atoms, delta=delta, indices=self.indices, nfree=2)
        self.vdata = VibrationsData.from_2d(atoms, H,indices=self.indices)
        self.vib.H = H
        self.vib._vibrations = self.vdata
        return True
        
    def ideal_button_click(self,b):
        atoms = self.latoms[0]
        self.out_box.clear_output()
        with self.out_box: 
            if not self.set_vib(atoms):
                return 
            if self.indices is not None:
                if  self.indices != len(atoms):
                    print("Vibrational info is for Harmonic!" )
                    return
            vib = self.vib
            vib_energies = vib.get_energies() 
            
            if sum(np.iscomplex(vib_energies[6:])) > 0:
                print( sum(np.iscomplex(vib_energies[6:])) ,"Warn!Imaginary frequencies are found. change to 1e-4")
                vib_energies[6:] = np.where(np.iscomplex(vib_energies[6:]),1e-4,vib_energies[6:])      
            geometry = check_linearity(atoms)
            symmetrynumber = check_symmetry_number(atoms)        
            print(geometry, symmetrynumber)
            thermo = IdealGasThermo(vib_energies=vib_energies,
                                    potentialenergy=self.epot,
                                    atoms=atoms.copy(),
                                    geometry=geometry,
                                    symmetrynumber= symmetrynumber,spin=0)

            t= self.temp_box.value
            p= self.press_box.value *1000
            thermo.get_gibbs_energy(t, p , verbose=True)
            enthalpy = []
            entropy = []
            gibbs = []
            ts = list(range(10,1010,10))
            for t in ts:
                enthalpy.append(thermo.get_enthalpy (t    , verbose=False))
                entropy.append (thermo.get_entropy  (t, p , verbose=False))
                gibbs.append(thermo.get_gibbs_energy(t, p , verbose=False))
            df = pd.DataFrame( ts,columns=["temp"] )
            df["enthalpy"] = enthalpy
            df["entropy"]  = entropy
            df["gibbs"]  = gibbs
            df["p"] = p
            self.df = df
            fig = thermo_plot(df)
            fig.show()

    def harmo_button_click(self,b):
        atoms = self.latoms[0]
        self.out_box.clear_output()
        with self.out_box: 
            if not self.set_vib(atoms):
                return       
            vib = self.vib
            vib_energies = vib.get_energies() 
            print("Imaginary", sum(np.iscomplex(vib_energies))) 
            real_vib_e = vib_energies[vib_energies.imag == 0].real
            thermo = HarmonicThermo(real_vib_e, potentialenergy=0)
            t= self.temp_box.value
            thermo.get_helmholtz_energy(temperature=t)
            internal_energy = []
            entropy = []
            helmholtz_energy = []
            ts = list(range(10,1010,10))
            for t in ts:
                internal_energy.append(thermo.get_internal_energy(t  , verbose=False))
                entropy.append(thermo.get_entropy(  t,  verbose=False))
                helmholtz_energy.append(thermo.get_helmholtz_energy(t,  verbose=False))            

  
            df = pd.DataFrame( ts,columns=["temp"] )
            df["internal_energy"] = internal_energy
            df["entropy"]  = entropy
            df["helmholtz_energy"]  = helmholtz_energy
            self.df = df
            fig = thermo_plot(df)
            fig.show()
            
    def __repr__(self):
        display(self.view)
        return "thermo_maker" 

def thermo_plot(df,h = 360 ,w = 360 ):
    cols = df.columns.to_list()
    if "p" in cols:
        print("IdealGasThermo" ,df.p[0] * 0.001 ,"kPa")
    else:
        print("HarmonicThermo  epot set 0.0")
    cols =  [n for n in cols if n not in ["temp","p"]]
    fig = make_subplots(rows=2, cols=1,)
    for i,name in enumerate(cols): 
        if name ==  "entropy":
            row = 2
        else:
            row = 1
        fig.append_trace(go.Scatter(x = df.temp,  y = df[name] ,name = name), row=row, col=1)
    fig.update_yaxes(title_text=None)
    fig.update_xaxes(title_text="template ,K",row=2)
    fig.update_layout(height=h, width=w,
                      margin=dict(l=20, r=20, t=40, b=5),
                      legend=dict(y = 1.1 ,x =0,orientation='h')
                     )
    return fig

def get_inertia_tensor(atoms: Atoms) -> np.ndarray:
    """Calculate the inertia tensor of molecule.
    Args:
        atoms (Atoms): The input structure.
    Returns:
        np.ndarray : The inertia tensor.
    """
    com = atoms.get_center_of_mass()
    r = atoms.positions - com
    x = r[:, 0]
    y = r[:, 1]
    z = r[:, 2]
    m = atoms.get_masses()
    Ixx = np.sum(m * (y ** 2 + z ** 2))
    Iyy = np.sum(m * (x ** 2 + z ** 2))
    Izz = np.sum(m * (x ** 2 + y ** 2))
    Ixy = -np.sum(m * x * y)
    Iyz = -np.sum(m * y * z)
    Ixz = -np.sum(m * x * z)
    return np.array(
        [
            [Ixx, Ixy, Ixz],
            [Ixy, Iyy, Iyz],
            [Ixz, Iyz, Izz],
        ]
    )


def get_total_inertia(atoms: Atoms) -> float:
    """Calculate the moment of inertia of molecule.

    Args:
        atoms (Atoms): The input structure.
    Returns:
        float : The momenta of inertia.
    """
    com = atoms.get_center_of_mass()
    r = np.linalg.norm(atoms.positions - com, axis=1)
    total_inertia: float = np.sum(atoms.get_masses() * r ** 2)
    return total_inertia


def check_linearity(atoms: Atoms, eigen_tolerance: float = 0.0001) -> str:
    """Check the molecule is 'Linear', 'Nonlinear' or 'monatomic'.

    Args:
        atoms (Atoms): The input structure.
        eigen_tolerance (float, optional): Tolerance of eigen value of inertia tensor when
            counting the number of zero elements. Defaults to 0.0001.
    Returns:
        str : The linearity of the molecule.
    """
    if len(atoms) == 1:
        return "monatomic"
    inertia_tensor = get_inertia_tensor(atoms) / get_total_inertia(atoms)
    w, v = np.linalg.eigh(inertia_tensor)
    if np.sum(w < eigen_tolerance) == 1:
        return "linear"
    elif np.sum(w < eigen_tolerance) == 0:
        return "nonlinear"
    else:
        raise ValueError("Cannot identify the linearity of the molecule.")


def check_symmetry_number(atoms: Atoms) -> int:
    """Calculate the symmetry number of molecule.

    Args:
        atoms (Atoms): The input structure.
    Returns:
        int : The symmetry number.
    """
    mol = AseAtomsAdaptor().get_molecule(atoms)
    point_group = PointGroupAnalyzer(mol).get_pointgroup()
    symbol = point_group.__str__()
    if symbol in ["C1", "Ci", "Cs", "C*v"]:
        return 1
    elif symbol in ["D*h"]:
        return 2
    elif symbol[0] == "S":
        return int(symbol[1]) // 2
    elif symbol[0] == "C":
        return int(symbol[1])
    elif symbol[0] == "D":
        return int(symbol[1]) * 2
    elif symbol[0] == "T":
        return 12
    elif symbol in ["Oh"]:
        return 24
    elif symbol in ["Ih"]:
        return 60
    else:
        return 1
    
def my_2d_hess(atoms,delta=0.01, indices=None,thread_n =12,calculator = None):
    ###ヘシアン並列計算
    model_version = calculator.estimator.model_version
    calc_mode  =  calculator.estimator.calc_mode   
    if indices is  None:
        indices= list(range(len(atoms)))
    n = len(indices)
    # 対象原子のxyz方向にd ,-d移動した座標をvibpos に格納
    p0 = atoms.positions.copy()
    vibpos = []
    for i in indices:
        for xyz in [0,1,2]:
            for plusminus in [-1,1]:
                p = p0.copy()
                p[i][xyz]  +=  delta * plusminus
                vibpos  += [p]
    #スレッド数だけcalcを設定したatomsオブジェクトを準備 ・・　vibimages
    vibimages=[]
    for i in range(thread_n):
        image = atoms.copy()
        estimator = Estimator(calc_mode=calc_mode,model_version=model_version) # For Matlantis #202210
        calculator = ASECalculator(estimator)
        image.calc = calculator
        vibimages +=[image]
    #対象原子あたり6構造計算するので6*n回のget_forcesが必要 
    #forces は　len(atoms),3の次元を持つので先に入れ物を準備
    forces = np.empty((6 * n , len(atoms) , 3))
    #スレッド数ごとに 準備した構造のforceを計算
    def run(image, forces):
        forces[:] = image.get_forces()
    start_n,end_n  = 0 , 0
    while end_n < 6*n :
        end_n = min( 6*n ,start_n + thread_n)
        for  i , no in enumerate( range(start_n,end_n ,1) ):
            vibimages[i].positions =  vibpos[no]    
        threads = [threading.Thread(target=run,
                                    args=(vibimages[i%thread_n],forces[i:i+1]))
                   for i in range(start_n,end_n ,1)]
        for thread in threads:        thread.start()
        for thread in threads:        thread.join()
        start_n += thread_n
    #forces 結果より　Hessianを計算
    H = np.empty((3 * n, 3 * n ))
    r = 0
    for i in range(n):
        for dirc in [0,1,2]:            
            H[i*3 + dirc] = 0.25 * (forces[r] - forces[r+1])[indices].ravel() /delta
            r += 2
    H += H.copy().T
    del vibimages
    gc.collect()
    return H

class Opter:
    def __init__(self , latoms = None , set_func= None , undo_button = None ,calculator = None):
        self.latoms  = latoms
        self.atoms = self.latoms[0]
        self.set_func = set_func
        self.calculator = calculator 
        self.fmax_box    = FloatText(description='fmax'    ,value=0.01 ,step = 0.001, layout={"width" : "120px"} ,style =  {"description_width": "55px"})
        self.maxstep_box = FloatText(description='maxsteps',value=0.1  ,step = 0.001, layout={"width" : "120px"} ,style =  {"description_width": "55px"})
        self.steps_box   = IntText  (description='steps'   ,value=100  ,step = 20,    layout={"width" : "120px"} ,style =  {"description_width": "55px"})
        fixlabel = Label("FIX Atoms indices")
        self.indices  = Indices_setter(latoms = latoms)
        self.opt_cell_check = Checkbox(description='opt_cell' , indent=False, layout={"width" : "80px"} )
        opt_button = Button(description='opt',layout ={"width" : "130px"} ) 
        opt_button.on_click(self.opt_button_click)
        readconst_button = Button(description='read const',layout ={"width" : "140px"} ) 
        readconst_button.on_click(self.readconst_button_click)        
        fixbondlabel = Label("FIX Bond")
        self.fixbonda0 = IntText  (description='a0'   ,value=0 ,    layout={"width" : "110px"} ,style =  {"description_width": "25px"})
        self.fixbonda1 = IntText  (description='a1'   ,value=0 ,    layout={"width" : "110px"} ,style =  {"description_width": "25px"})
        
        self.out_box=Output(layout=Layout(overflow_y='scroll', overflow_x='hidden', height='250px' ,width = "385px"))        
        self.view = VBox ([
            HBox([self.fmax_box,self.maxstep_box,self.steps_box]),  
            HBox([opt_button,self.opt_cell_check]),
            HBox([fixlabel,readconst_button]),
            self.indices.view,
            HBox([fixbondlabel,self.fixbonda0,self.fixbonda1]),
            self.out_box
        ])

    def readconst_button_click(self,b):
        atoms = self.latoms[0]
        if len(atoms.constraints) > 0:
            c = atoms.constraints[0]
            if type(c) is FixAtoms: 
                self.indices.selected_atoms_textarea.value = ", ".join(map(str, c.index.tolist()))
        
    def opt_button_click(self,b):
        atoms = self.latoms[0]
        self.atoms = atoms
        name = atoms.get_chemical_formula() + "_" +nowtime()
        self.logfile = "./output/opt/log/opt_" + name + ".txt"
        self.ciffile  ="./output/opt/" + name + ".cif"
        if os.path.isfile(self.logfile):
            #既にある、クリック2回受付した可能性
            return
        
        if atoms.calc is None or issubclass ( spc , type(atoms.calc) ) :  
            atoms.calc = self.calculator
             
        self.out_box.clear_output()
        with self.out_box: 
            constraint_atom_indices = self.indices.get_selected_atom_indices()
            if len(constraint_atom_indices) >0:
                atoms.set_constraint(FixAtoms(indices=constraint_atom_indices))     
                print("Fix atoms count: ", len( constraint_atom_indices))    
            else :
                atoms.set_constraint()   
            
            if self.fixbonda0.value !=  self.fixbonda1.value:
                fixbond = FixBondLengths([[self.fixbonda0.value,self.fixbonda1.value]])
                atoms.set_constraint( atoms.constraints + [fixbond])
                print("fixbond",self.fixbonda0.value,self.fixbonda1.value ,"l=", fixbond.initialize_bond_lengths(atoms)[0])
                     
            if self.opt_cell_check.value and sum(atoms.pbc) == 3 :
                if self.maxstep_box.value >= 0.1: self.maxstep_box.value = 0.005  
                atoms = opt_cell_size(atoms,
                                      fmax    = self.fmax_box.value ,
                                      steps   = self.steps_box.value , 
                                      maxstep = self.maxstep_box.value,
                                      logfile = self.logfile ) 
            else :
                self.opt_cell_check.value = False
                if self.maxstep_box.value <= 0.005: self.maxstep_box.value = 0.1  
                atoms =  myopt(atoms,
                               fmax  = self.fmax_box.value ,
                               steps = self.steps_box.value ,
                               maxstep= self.maxstep_box.value,
                               logfile = self.logfile)
            
            atoms.write(self.ciffile)
            self.set_func([atoms])
            fig = opt_plot(self.logfile)
            fig.show()
            print(self.logfile)
             
    def __repr__(self):
        display(self.view)
        return "Opter"

def opt_plot(logfile ,h = 220 ,w = 360 ):
    df = pd.read_csv( logfile , delim_whitespace=True, skiprows=2, header=None, usecols=[1, 3, 4])
    df.columns = ['Step', 'energy', 'fmax']
    df['energy'] = df['energy'].str.replace('*', '', regex=False).astype(float)
    fig = make_subplots(rows=2, cols=1,)
    fig.append_trace(go.Scatter(x = df.Step,  y = df.energy ,name = 'epot eV'), row=1, col=1)
    fig.append_trace(go.Scatter(x = df.Step , y = df.fmax   ,name = 'fmax'   ), row=2, col=1)
    fig.update_yaxes(title_text=None)
    fig.update_xaxes(title_text="Steps",row=2)
    fig.update_layout(height=h, width=w,
                      margin=dict(l=5, r=5, t=5, b=5),
                      legend=dict(x=1, y=1, xanchor='right', yanchor='top')
                     )
    return fig

def myopt(atoms, fmax=0.01, steps=200, maxstep=0.04, optimizer="LBFGS", logfile=None, trajectory=None):  
    print(f"step:{0:5}" ,end =" ")
    print(f"Epot: {atoms.get_potential_energy():.6f} fmax: {np.linalg.norm(atoms.get_forces(), axis=1, ord=2).max():.6f}")
    if optimizer=="LBFGS":
        opt = LBFGS(atoms, maxstep=maxstep, logfile=logfile, trajectory=trajectory)
    elif optimizer=="FIRE":
        opt = FIRE(atoms, maxstep=maxstep,  logfile=logfile, trajectory=trajectory)
    else:
        raise ValueError(f"optimizer_{optimizer} is not LBFGS or FIRE")
    opt.run(fmax=fmax, steps=steps)
    print(f"step:{opt.nsteps:5}" ,end =" ")
    print(f"Epot: {atoms.get_potential_energy():.6f} fmax: {get_fmax(atoms):.6f}")
    return atoms

def opt_cell_size(atoms, fmax=0.01, steps=200, maxstep=0.001, logfile=None):
    # 対称性保存
    atoms.set_constraint([FixSymmetry(atoms, adjust_cell=True, adjust_positions=True)])
    print(f"step:{0:5}" ,end =" ")
    print(f"Epot: {atoms.get_potential_energy():.6f} fmax: {np.linalg.norm(atoms.get_forces(), axis=1, ord=2).max():.6f}")
    lx,ly,gm   = atoms.cell.cellpar()[[0,1,5]]
    print (f"AREA:      {lx:.2f} A x {ly:.2f} A  GAMMA:  {gm:.2f} deg.")
    print (f"THICKNESS  {np.max (atoms.get_positions ()[:, 2]) - np.min (atoms.get_positions ()[:, 2]):.2f} A")
    
    ecf = ExpCellFilter(atoms)
    opt = LBFGS(ecf, maxstep=maxstep, logfile=logfile)
    opt.run(fmax=fmax, steps=steps)
    print(f"step:{opt.nsteps:5}" ,end =" ")
    print(f"Epot: {atoms.get_potential_energy():.6f} fmax: {np.linalg.norm(atoms.get_forces(), axis=1, ord=2).max():.6f}")
    lx,ly,gm   = atoms.cell.cellpar()[[0,1,5]]
    print (f"AREA:      {lx:.2f} A x {ly:.2f} A  GAMMA:  {gm:.2f} deg.")
    print (f"THICKNESS  {np.max (atoms.get_positions ()[:, 2]) - np.min (atoms.get_positions ()[:, 2]):.2f} A")
    atoms.set_constraint()
    return atoms


class Liquid_maker:
    def __init__(self , latoms_list = None , set_func= None , undo_button = None, calculator = None):
        self.latoms_list = latoms_list 
        self.set_func = set_func
        self.calculator = calculator
        self.generator = None
        l = Label("to make Liquid from atoms")
        self.atoms0b = Dropdown( options= range(20) ,description='Atoms0'  ,layout={"width" : "180px"} ) 
        self.atoms1b = Dropdown( options= range(20) ,description='Atoms1'  ,layout={"width" : "180px"} )
        self.a0n = IntText ( description='*N' ,value = 10 ,layout={"width" : "180px"} ) 
        self.a1n = IntText ( description='*N' ,layout={"width" : "180px"} ) 
        self. density= FloatText(step = 0.05, description='density'  ,value = 0.6 ) 
        make_liq_button = Button(description='liq make',layout = {"width" : "100px"}) 
        make_liq_button.on_click(self.make_liq_button_click)
        self.out_box=Output(layout=Layout(overflow_y='scroll', overflow_x='hidden', height='300px' ,width = "385px"))        

        self.view  = VBox ( [l,self.atoms0b,self.a0n,self.atoms1b,
                             self.a1n ,self.density, 
                             make_liq_button,self.out_box])

    def make_liq_button_click(self,b):
        if "LiquidGenerator" not in locals():
            module_path = os.path.dirname(os.path.abspath(__file__))
            sys.path.append(module_path)  # インポートパスに追加
            from liquidgenerator import LiquidGenerator
        
        self.out_box.clear_output()
        with self.out_box:
            latoms_list = self.latoms_list
            a0id = self.atoms0b.index
            a1id = self.atoms1b.index
            atoms0 = latoms_list[a0id][0].copy()
            atoms1 = latoms_list[a1id][0].copy()
            atoms_list =  [atoms0] * self.a0n.value + [atoms1] * self.a1n.value
            c = sum ([len(a) for a in  atoms_list ] )
            print("all atoms count" ,c)
            if c > 1000:
                print("stop calc. count over 1000!")
                return
            self.generator = LiquidGenerator(atoms_list, density= self.density.value )
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=UserWarning)
                self.generator.fit(epochs=100, lr=0.1)
            atoms_mixture = self.generator.to_atoms()
            self.set_func( [atoms_mixture] )
                    
    def __repr__(self):
        display(self.view)
        return "Liquid_maker"
    

class Calc_setter:
    def __init__(self, calculator = None ,ngl = None):
        if calculator is None:
            estimator = Estimator()
            calculator = ASECalculator(estimator)
        self.ngl = ngl
        
        versions = Estimator().available_models[::-1]
        modes    = [mode.name for mode in EstimatorCalcMode]
        methods  = [method.name for method in EstimatorMethodType]
        l  =Label("PFP setting")
        self.versionbox =  widgets.Dropdown(description="version",options=versions,index = find_index(versions,calculator.estimator.model_version) )
        self.modebox    =  widgets.Dropdown(description="mode"   ,options=modes   ,index = find_index(modes   ,calculator.estimator.calc_mode) )
        self.methodbox  =  widgets.Dropdown(description="method" ,options=methods ,index = find_index(methods ,calculator.estimator.method_type) )
        self.outbox = Output( layout =  {"width" :'350px'}  )
        self.calculator = calculator
        set_button = Button(description='SET',layout =  {"width" :'60px'} )     
        set_button.on_click (self.set_button_click) 
        self.view = VBox([l,self.versionbox ,self.modebox,self.methodbox ,set_button,self.outbox],layout =  {"width" :'370px'} )
        with self.outbox:
            print ("from  " ,calculator.estimator.model_version , calculator.estimator.calc_mode  , calculator.estimator.method_type.name)
        
    def set_button_click(self,b):
        self.outbox.clear_output()
        self.calculator.estimator.model_version = self.versionbox.value
        self.calculator.estimator.calc_mode     = self.modebox.value
        self.calculator.estimator.method_type   = self.methodbox.value
        self.calculator.reset()
        with self.outbox:
            print ("set to" ,self.calculator.estimator.model_version , self.calculator.estimator.calc_mode  , self.calculator.estimator.method_type.name)

        if self.ngl is not None:
            #NGLで結果が変わるか確認する。（引数でNGL渡された場合）
            self.ngl.check_calc.value = True
            self.ngl.show_calc()

###UTIL functions ######     
def nowtime() :
    return f"{datetime.datetime.now(pytz.timezone('Asia/Tokyo')):%Y%m%d%H%M%S}"

def get_fmax(atoms):
    return np.linalg.norm(atoms.get_forces(), axis=1, ord=2).max()

def find_index(lst, item):
    ###for calclator setting
    if type(item) is not str:
        if hasattr(item,"name"):
            item = item.name
        else:
            item = "noname"
    mydict = {value.upper(): index for index, value in enumerate(lst)}
    return mydict.get(item.upper(), 0)     

def clear_output_wo_edges(out_box, head=4, tail=1):
    lines = [output['text'] for output in out_box.outputs if 'text' in output]
    if len(lines) > 0:
        lines = lines[0].split("\n")
        lines = [ line for line in lines if line != ""]
        lines_to_keep = lines[:4] + lines[-1:]
        out_box.clear_output()
        with out_box:
            for line in lines_to_keep:
                print(line)
                
def get_num(text):
    if text is None:
        return 0
    match = re.search(r'\d+',text)
    number = int(match.group()) if match else 0
    return number

def energy_fmax_plot(es = None,h = 400 ,w = 360):
    df = pd.DataFrame( es,columns=["energy","fmax"] )
    fig = make_subplots(rows=2, cols=1)
    fig.append_trace(go.Scatter(x = df.index,  y = df.energy ,name = 'epot eV'),row=1, col=1)
    fig.append_trace(go.Scatter(x = df.index , y = df.fmax   ,name = 'fmax'   ),row=2, col=1)
    fig.update_yaxes(title_text=None)
    fig.update_xaxes(title_text="index",row=2)
    fig.update_layout(height=h, width=w,margin=dict(l=5, r=5, t=5, b=0))
    return fig

def get_traj_energy_fmax_parallel( latoms , calculator= None , thread_n =12, ):
    if calculator is None:
        estimator = Estimator()
        calculator = ASECalculator(estimator)
    model_version = calculator.estimator.model_version
    calc_mode     = calculator.estimator.calc_mode   
    n = len(latoms)
    #スレッド数を準備 ・calcs
    calcs=[]
    for i in range(thread_n):
        estimator = Estimator(calc_mode=calc_mode,model_version=model_version) 
        calculator = ASECalculator(estimator)
        calcs +=[calculator]
    energy_fmaxs = np.empty((n ,2))
    #スレッド数ごとに energy fmaxを計算
    def run(atoms, energy_fmax, calcn):
        atoms =atoms.copy()
        atoms.calc = calcs[calcn]
        energy_fmax[:] = [atoms.get_potential_energy() , np.linalg.norm(atoms.get_forces(), axis=1, ord=2).max()]
    start_n,end_n  = 0 , 0
    while end_n < n :
        end_n = min( n ,start_n + thread_n) 
        threads = [threading.Thread(target=run,
                                    args=( latoms[i] , energy_fmaxs[i:i+1] , i%thread_n))
                   for i in range(start_n,end_n ,1)]
        for thread in threads:        thread.start()
        for thread in threads:        thread.join()
        start_n += thread_n
    del calcs
    return energy_fmaxs
###### 

class Matviewer:
    def __init__(self,calculator = None ,load = True):
        
        if calculator is None:
            estimator = Estimator()
            calculator = ASECalculator(estimator)
        self.calculator = calculator
        outpaths = ["./output","./output/opt","./output/opt/log","./output/save",
                    "./output/neb","./output/neb/log","./output/md","./output/vib","./output/thermo"]
        for p in outpaths:
            os.makedirs(p, exist_ok=True)
        
        ### COMMON variables ###### 
        self.temp_atoms_list = []
        self.latoms = [Atoms("Au",[[0,0,0]])]
        self.latoms_list =[self.latoms.copy()]
        if load:
            self.load_function()
                    
        ###  MENU ####
        self.add_menu = Dropdown(options=['ADD', 'File','SMILES or Draw',"Crystal","MOF"],layout = {"width": "140px"})
        self.add_menu.observe(lambda change: self.add_function(change.new) if change['name'] == 'value' else None, 'value')
        self.edit_menu = Dropdown( options=['EDIT',"Move","Utiledit","Repeat","Cell", "Surface", "Attach","Liquid","Traj" ],layout = {"width": "140px"})
        self.edit_menu.observe(lambda change: self.edit_function(change.new) if change['name'] == 'value' else None, 'value')
        self.calc_menu = Dropdown( options=['CALC','Setting','OPT','NEB', 'VIB','THERMO', 'MD_NVT'],layout = {"width": "140px"} )
        self.calc_menu.observe(lambda change: self.calc_function(change.new) if change['name'] == 'value' else None, 'value')
                
        self.save_button = Button(description='Save',  layout =  Layout(height='30px',width='50px') )
        self.save_button.on_click(self.save_function)   
        
        self.app_output  = Output(layout= {"height":'1px',"width":'1px'} )#for back ground 
        menubar = HBox([self.add_menu, self.edit_menu,self.calc_menu,self.save_button , self.app_output ])

        ### Overall view Layout
        self.leftctlbox  = Box (layout= {"height":'500px',"width":'400px'} )  
        self.rightctlbox = VBox(layout= {"height":'500px',"width":'900px'} ) #ディスプレイ初めに設定した値が後を引くので少し大きめ   
        self.view = VBox ([menubar,  HBox ([self.leftctlbox ,self.rightctlbox])])      

        ### RIGHT_Atoms_Setter and Selectors ###
        self.add_n_button = Button(description='Add#',  layout =  Layout(height='30px',width='60px') )
        self.add_n_button.on_click(self.add_n_function)
        
        self.add_nn_button = Button(description='Add##',  layout =  Layout(height='30px',width='60px') )
        self.add_nn_button.on_click(self.add_nn_function)
        
        self.selector0 = Dropdown(options=[0], layout = {"width": "62px"}, index=0,description="#",style={"description_width": "8px"} )        
        self.selector0.observe(self.selector0_selected, names='value')
        self.selector0.options = list(range(len(self.latoms_list)))
        
        self.selector1=IntSlider(value=1, min=0,max=0,step=1, layout = Layout(height='30px',width='170px') )
        self.selector1.observe(self.selector1_selected,names='value' )
        self.del_n_button = Button(description='Del#',  layout =  Layout(height='30px',width='57px') )
        self.del_n_button.on_click(self.del_n_function)        
        self.del_nn_button = Button(description='Del##',  layout =  Layout(height='30px',width='57px') )
        self.del_nn_button.on_click(self.del_nn_function)
        self.plot_button = Button(description='Plot',  layout =  Layout(height='30px',width='50px') )
        self.plot_button.on_click(self.show_plot) 
        self.selector = HBox (  layout= {"height": "40px"})
        self.set_selector_view()
        
        #### RIGHT NGL #######
        self.nglbox = Nglbox(calculator=self.calculator)
        self.nglbox.check_index.value = True
        self.nglbox.check_pbc.value   = True        
        self.rightctlbox.children =[self.selector, self.nglbox.viewwithcheck] 
        self.nglbox.update_structre(self.latoms[0])
        
        ####common###
        self.undo_button = Button(description='Undo',layout = {"width": "140px"})
        self.undo_button.on_click(self.undo_button_click)
        self.indices_setter  = Indices_setter(latoms = self.latoms)
        #### Left Ctls #######
        self.crystalsearch = Crystalsearch_mini(set_func=self.set_func)
        self.mofsearch     = Mofsearch_mini(set_func=self.set_func)
        self.fileselector = File_seceltor(set_func=self.set_func)
        self.repeat_ctl = Repeater     (set_func=self.set_func, latoms=self.latoms ,undo_button=self.undo_button)
        self.cell_ctl   = Cell_setter  (set_func=self.set_func, latoms=self.latoms ,undo_button=self.undo_button)
        self.move_ctl   = Mover        (set_func=self.set_func, latoms=self.latoms ,undo_button=self.undo_button)
        self.util_ctl   = Util_editor  (set_func=self.set_func, latoms=self.latoms ,latoms_list=self.latoms_list,undo_button=self.undo_button)
        self.traj_ctl   = Traj_setter  (reset_selector_view_func=self.reset_selector_view, latoms_list=self.latoms_list )
        

        self.sur_ctl    = Surface_maker(set_func=self.set_func, latoms=self.latoms ,undo_button=self.undo_button)
        self.opt_ctl    = Opter        (set_func=self.set_func, latoms=self.latoms ,undo_button=self.undo_button,calculator = self.calculator)
        self.nvt_ctl    = NVT_maker    (set_func=self.set_func, latoms=self.latoms ,calculator = self.calculator)

        self.attach_ctl = Attach_maker (set_func=self.set_func, latoms_list=self.latoms_list ,undo_button=self.undo_button,calculator = self.calculator)
        self.liquid_ctl = Liquid_maker (set_func=self.set_func, latoms_list=self.latoms_list ,undo_button=self.undo_button,calculator = self.calculator)
        
        self.neb_ctl    = Neb_maker    (set_func=self.set_func, latoms_list=self.latoms_list ,calculator = self.calculator)
        self.vib_ctl    = Vib_maker    (set_func=self.set_func, latoms=self.latoms ,calculator = self.calculator)
        self.thermo_ctl = Thermo_maker (set_func=self.set_func, latoms=self.latoms ,calculator = self.calculator)

        self.pltoutput = Output(layout = {"height":"350px","width":"390px"})
        self.jsme = JSMEBox(set_func=self.set_func)
        self.calcset = Calc_setter(self.calculator ,ngl = self.nglbox )
        self.display_set_ini()
        self.set_ctl = VBox([self.calcset.view,self.dispctl],layout =  {"width" :'380px'} )
        self.leftctlbox.children = [self.set_ctl]
        
    def undo_button_click(self,b):
        if len(self.temp_atoms_list) > 0:
            poplatoms =self.temp_atoms_list.pop()
            self.latoms_list[self.selector0.index][self.selector1.value] = poplatoms[0]
            self.latoms[0] =  poplatoms[0]
            self.atoms =  poplatoms[0]
            self.nglbox.update_structre(poplatoms[0])
            
    ######   menu    ####################################
    def add_function(self,choice):
        if choice == 'File':          self.leftctlbox.children = [self.fileselector.view]
        if choice == 'SMILES or Draw':self.leftctlbox.children = [self.jsme.view]
        if choice == 'Crystal':       self.leftctlbox.children = [self.crystalsearch.view]
        if choice == 'MOF':       self.leftctlbox.children = [self.mofsearch.view]
        self.calc_menu.index = 0
        self.edit_menu.index = 0

    def edit_function(self,choice):
        if choice == 'Move'   : self.leftctlbox.children = [self.move_ctl.view]
        if choice =='Utiledit': self.leftctlbox.children = [self.util_ctl.view]
        if choice == 'Repeat' : self.leftctlbox.children = [self.repeat_ctl.view]
        if choice == 'Cell'   : self.leftctlbox.children = [self.cell_ctl.view]
        if choice == 'Surface': self.leftctlbox.children = [self.sur_ctl.view]
        if choice == 'Attach' : self.leftctlbox.children = [self.attach_ctl.view]
        if choice == "Liquid" : self.leftctlbox.children = [self.liquid_ctl.view]
        if choice == "Traj"   : self.leftctlbox.children = [self.traj_ctl.view]
        
        self.add_menu.index = 0
        self.calc_menu.index = 0

    def calc_function(self,choice):
        if choice == 'Setting': self.leftctlbox.children = [self.set_ctl]
        if choice == 'OPT'    : self.leftctlbox.children = [self.opt_ctl.view]
        if choice == 'NEB'    : self.leftctlbox.children = [self.neb_ctl.view]
        if choice == 'VIB'    : self.leftctlbox.children = [self.vib_ctl.view]
        if choice == 'THERMO' : self.leftctlbox.children = [self.thermo_ctl.view]
        if choice == 'MD_NVT' : self.leftctlbox.children = [self.nvt_ctl.view]
        self.add_menu.index = 0
        self.edit_menu.index = 0

    def save_function(self,change):
        directory = './output/save/'
        zip_filename   =  directory + nowtime() +'.zip'        
        traj_files = sorted(glob.glob(os.path.join(directory, '*.traj')), 
                            key=lambda x: int(''.join(filter(str.isdigit, x))))
        
        if len(traj_files) > 0:
            with zipfile.ZipFile(zip_filename, 'w') as zipf:
                for file_path in traj_files:
                    zipf.write(file_path, os.path.basename(file_path))
                    os.remove(file_path)     
        for  i,traj in enumerate(self.latoms_list):
            ase.io.write(  directory  + str(i) + '.traj' , traj)
            
    def load_function(self):
        directory = './output/save/'     
        traj_files = sorted(glob.glob(os.path.join(directory, '*.traj')), 
                            key=lambda x: int(''.join(filter(str.isdigit, x))))
        if len(traj_files) > 0:
            self.latoms_list  = [ase.io.read(traj,index=":") for traj in traj_files]
        else :
            self.latoms_list = [[Atoms("Au",[[0,0,0]])]]
        self.latoms = self.latoms_list[0].copy()
        self.atoms  = self.latoms[0]
                        
    def set_selector_view(self):
        self.selector0.unobserve(self.selector0_selected, names='value')
        self.selector1.unobserve(self.selector1_selected, names='value')
        s0len = len(self.latoms_list)
        if s0len != len(self.selector0.options) :  self.selector0.options  =list(range(s0len))
        s1max = len(self.latoms_list[self.selector0.index]) -1
        if s1max !=  self.selector1.max: self.selector1.max = s1max
        self.selector0.observe(self.selector0_selected, names='value')
        self.selector1.observe(self.selector1_selected, names='value')
        
        selectmenu = [self.add_n_button, self.add_nn_button , self.selector0,
                      self.selector1,self.plot_button,self.del_n_button ,self.del_nn_button]
        if len(self.latoms_list[self.selector0.index]) == 1:
            selectmenu.remove(self.selector1)
            selectmenu.remove(self.plot_button)
            selectmenu.remove(self.del_nn_button)
        if len(self.latoms_list) == 1:
            selectmenu.remove(self.del_n_button)
        if len(self.selector.children) != len(selectmenu):
            self.selector.children = selectmenu
            
    def reset_selector_view(self):
        self.set_selector_view()
        self.atoms = self.latoms_list[self.selector0.value] [self.selector1.value]
        self.latoms[0] = self.atoms
        self.nglbox.update_structre(self.atoms)
        
    def add_n_function(self,change): 
        atoms = self.latoms_list[self.selector0.index][self.selector1.value]
        self.latoms_list.append( [atoms.copy()])
        self.set_selector_view()
        self.selector0.value = len(self.latoms_list) - 1
        self.atoms = self.latoms_list[self.selector0.value] [self.selector1.value]
        self.latoms[0] = self.atoms

    def add_nn_function(self,change): 
        atoms = self.latoms_list[self.selector0.index][self.selector1.value]
        self.latoms_list[self.selector0.index].append( atoms.copy())
        self.set_selector_view()
        self.selector1.value = self.selector1.max
        self.atoms = self.latoms_list[self.selector0.value] [self.selector1.value]
        self.latoms[0] = self.atoms

    def del_n_function(self,change):  
        l = len(self.latoms_list) 
        if l > 1:        
            i = self.selector0.index
            self.latoms_list.pop(i)
            self.set_selector_view()
            self.selector0.value = min(i ,  l-2)
            self.reset_selector_view()
             
    def del_nn_function(self,change):  
        l = len(self.latoms_list[self.selector0.index])
        if l > 1:        
            i = self.selector1.value
            self.latoms_list[self.selector0.index].pop(i)
            self.set_selector_view()
            self.selector1.value = min(i ,  l-2)
            self.reset_selector_view()
         
    def selector0_selected(self,change): 
        self.selector1.unobserve(self.selector1_selected, names='value')
        self.selector1.value = 0
        self.selector1.observe(self.selector1_selected, names='value')
        self.reset_selector_view()

    def selector1_selected(self,change): 
        self.atoms = self.latoms_list[self.selector0.value] [self.selector1.value]
        self.latoms[0] = self.atoms
        self.nglbox.set_struct(self.atoms)
            
    def set_func(self,result_latoms ,addnewno = False):
        if len(result_latoms) > 0:
            if len(result_latoms[0]) > 20000:
                print("detect more than 20,000 atoms!")
                return               
            self.temp_atoms_list.append([self.latoms[0].copy()])
            self.latoms[0] = result_latoms[0]
            self.atoms = self.latoms[0]      
            
            if len(result_latoms) > 1 or addnewno:
                self.latoms_list += [result_latoms]
                self.set_selector_view()
                self.selector0.value = len(self.selector0.options) -1
            else:
                self.latoms_list[self.selector0.value][self.selector1.value] = result_latoms[0]
                if  np.array_equal(self.temp_atoms_list[-1][0].pbc , self.atoms.pbc):
                    self.nglbox.set_struct(self.atoms)
                else:                
                    self.nglbox.update_structre(result_latoms[0])            
    
            if len(self.temp_atoms_list) > 10:
                self.temp_atoms_list[:] = self.temp_atoms_list[-10:]  

    def show_plot(self,b):
        self.leftctlbox.children = [self.pltoutput]  
        self.calc_menu.index = 0
        self.add_menu.index = 0
        self.edit_menu.index = 0
        self.pltoutput.clear_output()
        with self.pltoutput:
            print( f"# {self.selector0.value}  images" )
            latoms = self.latoms_list[self.selector0.value]     
            self.ene_force = get_traj_energy_fmax_parallel(latoms ,self.calculator)    
            fig = energy_fmax_plot(self.ene_force)
            fig.show()
                   
    def display_set_ini(self):
        l  =Label("Display setting")
        w0 =get_num( self.rightctlbox.layout.width )
        h0 =get_num( self.rightctlbox.layout.height )
        self.wi = IntText ( description='width'  ,layout={"width" : "130px"}, style={"description_width": "40px"} ,value=w0,step=50) 
        self.hi = IntText ( description='height' ,layout={"width" : "130px"}, style={"description_width": "40px"} ,value=h0,step=50) 
        self.color_picker = ColorPicker(value='#3399AA', description='APP back')
        self.color_picker.observe(self.display_set_button_click) 
        set_button = Button(description='SET',layout =  {"width" :'60px'} )       
        set_button.on_click (self.display_set_button_click) 
        player = self.nglbox.view.player
        with self.app_output:
            display(player)
        def dummy_plot():
            with self.app_output:
                display(go.Figure( layout=dict(height=150, width=300, )))
        th = threading.Thread(target=dummy_plot)
        th.start()
        self.nglback = player.widget_background
        self.nglback. description='NGL back'
        player.widget_camera.value = "orthographic"
        full_button = Button(description='Full screen',layout ={"width" :'120px'} )       
        full_button.on_click (self.full_button_click) 
        small_button = Button(description='Smaller'   ,layout ={"width" :'120px'} )       
        small_button.on_click (self.small_button_click)
        big_button = Button(description='Bigger'      ,layout ={"width" :'120px'} )       
        big_button.on_click (self.big_button_click)
        self.dispctl = VBox([l, 
                             HBox([self.wi,self.hi,set_button ]),
                             full_button,
                             HBox([small_button,big_button]),
                             self.color_picker,self.nglback,player.widget_camera
                            ])
        self.app_output.clear_output()
        self.display_set_button_click("")
       
    def display_set_button_click(self,b):
        w = max (self.wi.value , 450)
        h = max (self.hi.value , 450)
        self.wi.value = w
        self.hi.value = h
        self.rightctlbox.layout.width   = str(w) + "px"
        self.nglbox.view.layout.width   = str(w-20) + "px"
        self.rightctlbox.layout.height  = str(h) + "px"
        self.leftctlbox.layout.height   = str(h) + "px"
        self.nglbox.view.layout.height  = str(h-150) + "px"
        self.view.layout.width = str(w + 430) + "px"
        self.app_output.clear_output()
        with self.app_output:
            display(HTML(f'<style> .widget-vbox {{background-color: {self.color_picker.value} ; padding: 2px }}</style>'))                     

        
    def full_button_click(self,b):        
        js_code = """
        var vboxElements = document.querySelectorAll('div .jp-CodeCell .jp-Cell-outputWrapper .widget-vbox');
        console.log("how many vbox? ",vboxElements.length);
        for (var i = 0; i < vboxElements.length; i++) {   
            var vbox = vboxElements[i];
            var button = vbox.querySelector('.widget-label[title="model_id"]');
            if (button) {
                console.log("hit my model", button);
                vbox.requestFullscreen();
                break; 
            } 
        }
        """
        self.add_menu.description_tooltip = self.add_menu.model_id
        js_code = js_code.replace("model_id",self.add_menu.model_id)
        with self.app_output:
            display(Javascript(js_code))
        if self.wi.value < 1000: self.wi.value += 200
        if self.hi.value < 700: self.hi.value  += 200
        self.display_set_button_click("")
        
    def small_button_click(self,change):
        self.wi.value  -= 100
        self.hi.value  -= 50
        self.display_set_button_click("")
        
    def big_button_click(self,change):
        self.wi.value  += 100
        self.hi.value  += 50
        self.display_set_button_click("")
        
    def __add__(self, other):
        if hasattr(other, "latoms_list"):
            self.latoms_list += other.latoms_list    
        elif hasattr(other, "positions"):
            self.latoms_list += [[other]]
        elif type(other) is list and hasattr(other[0], "positions"): 
            self.latoms_list += [other]
        else : return 
        self.set_selector_view()
        self.selector0.index   = len(self.selector0.options ) - 1    
        return 
    
    def __iadd__(self, other):
        self.__add__(other)
        return 

    def add_atoms(self,atoms):
        self.__add__(atoms)
        return 
                    
    def __repr__(self):
        display(self.view)
        return ""