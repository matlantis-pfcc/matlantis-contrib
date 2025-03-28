# import
import numpy as np
from math import pi, sqrt
import math, time, os, shutil, re, glob, pickle, collections
from IPython.display import clear_output
from pfp_api_client.pfp.calculators.ase_calculator import ASECalculator
from pfp_api_client.pfp.estimator import Estimator
import autode as ade
# ↓ autodEのインストール
# !pip install git+https://GitHub.com/duartegroup/autodE.git

# ase
import ase
from ase import Atom, Atoms, units
from ase.optimize import LBFGS,FIRE
from ase.build import add_adsorbate
from ase.constraints import ExpCellFilter, FixAtoms
from ase.visualize import view
from ase.io import read, write
from ase.data.pubchem import pubchem_atoms_search, pubchem_atoms_conformer_search
from ase import neighborlist
from ase.data import covalent_radii

# matlantis_features
from matlantis_features.ase_ext.optimize import FIRELBFGS

# NGL関連
from IPython.display import HTML #ipywidgetsからimportされるHTMLに上書きされるので場所注意
display(HTML('<style> .widget-vbox {background-color:transparent;padding: 2px}</style>'))

import nglview as nv
from typing import Dict, List, Optional
from ase.visualize.ngl import NGLDisplay
from IPython.display import display
from ipywidgets import (Accordion, Button, Checkbox, FloatSlider, FloatRangeSlider, GridspecLayout, HBox, VBox, Dropdown,
                        IntSlider, Label, Output, Text, Textarea, BoundedIntText, BoundedFloatText, HTML, Tab, Select, SelectMultiple, 
                        Layout, RadioButtons, Box, Image,ToggleButtons,widgets)
from traitlets import Bunch
from nglview.widget import NGLWidget
from pfcc_extras.visualize.ngl_utils import (add_force_shape, add_axes_shape, get_struct,
                                             save_image, update_tooltip_atoms)


# 205行目までAddEditorで使う関数
def smiles_to_ase_atoms(smiles):
    """smiles to ase atoms
    """    
    ade_mol = ade.Molecule(smiles=smiles)
    ase_atoms = ade_to_ase_atoms(ade_mol.atoms)
    return ase_atoms

def ade_to_ase_atoms(ade_atoms):
    """autode atoms to ase atoms
    """
    ele = np.array([atom.atomic_symbol for atom in ade_atoms])
    pos = np.array([atom.coord for atom in ade_atoms])
    return Atoms(ele, pos)

def f_neighbor_atom(mol, a_index):
    """結合原子のindexを配列で取得
    """
    # get symbols
    chemical_symbols = mol.get_chemical_symbols()
    # cuf off settings (default: natular cutoffs)
    cut_off = neighborlist.natural_cutoffs(mol)
    # neighborlist in ASE
    neighborList = neighborlist.NeighborList(cut_off, self_interaction=False, bothways=True)
    neighborList.update(mol)

    neighbor_atom = neighborList.get_neighbors(a_index)[0]
    return neighbor_atom

def get_covalent_radii(element_a, element_b):
    """ 指定した2つの原子の推定共有結合半径を出力する関数
    """
    # 元素の周期表番号を取得します。
    atomic_number_a = Atoms(element_a).numbers[0]
    atomic_number_b = Atoms(element_b).numbers[0]

    # covalent_radiiモジュールから共有結合半径を取得します。
    radius_a = covalent_radii[atomic_number_a]
    radius_b = covalent_radii[atomic_number_b]

    # 推定される共有結合距離は、2つの原子の共有結合半径の和です。
    estimated_bond_distance = radius_a + radius_b

    return estimated_bond_distance

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
    comppos = np.delete(positions,index,0)
    comppos = comppos[np.argsort(np.linalg.norm(comppos-positions[index] ,axis=1))[:50]]    
    mindis = [ min(np.linalg.norm(comppos-cand ,axis=1))  for cand in cands]#候補座標から最短原子までの距離
    mindis = np.array(mindis)
    return cands[np.argmax(mindis)]#最短が最長となる座標



def add_mole_editor(atoms_sbs,atoms_ads,indx_sbs,indx_ads,a_bond_length=None,a_rotate=None, a_replace=True):
    """ベンゼンからトルエンを作りたいとき、引数はベンゼン(残す分子),メタン(消す分子),ベンゼンのindex[残す,消す],メタンのindex[消す,残す], a_replace=Falseにするとatoms_sbsの構造を変更しない
    """
    if a_replace==True : 
        A1=atoms_sbs
    else :
        A1=atoms_sbs.copy()
    A2=atoms_ads.copy()
#ベクトル作成
    vec_sbs = A1[indx_sbs[0]].position - A1[indx_sbs[1]].position
    vec_ads = A2[indx_ads[0]].position - A2[indx_ads[1]].position
#内積を使って、角度thetaを作ってます
    theta=np.arccos(np.inner(vec_ads, vec_sbs)/np.linalg.norm(vec_ads)/np.linalg.norm(vec_sbs))*180/math.pi
    A2.rotate(a=theta,v=np.cross(vec_ads,vec_sbs),center=A2[indx_ads[1]].position)

#くっつける方を回す
#平行移動
    j=0
    j+=A1[indx_sbs[1]].position
    j-=A2[indx_ads[1]].position
    
    #結合距離を指定
    ##############################################################################
    if a_bond_length==None: pass
    else :
        unit_vector_sbs = vec_sbs/np.linalg.norm(vec_sbs)   #単位行列に変換
        j+=unit_vector_sbs*np.linalg.norm(vec_sbs)    #くっつける側の原子とくっつける原子の座標を合わせる
        j-=unit_vector_sbs*a_bond_length    #くっつける原子の座標を「単位ベクトル×結合距離」分だけ離す(上3行を1行にすることも可能)
    ##############################################################################
      
    tmp=A2.copy()
    for i in range(len(tmp)):
        tmp[i].position+=j
    A2=tmp.copy()

    #くっつける軸を回転させる
    ##############################################################################    
    if a_rotate==None : pass
    else :
        A2.rotate(a=a_rotate,v=vec_sbs,center=A2[indx_ads[1]].position)
    ##############################################################################

#後ろの方のインデックスを消す
    A1.pop(indx_sbs[1])
    A2.pop(indx_ads[0])
    output=A1
    for i in A2:
        output.append(i)
    return output


"""ここから下はadd_mole関数の引数を減らすために最近接原子を探す関数
"""
"""スラブ側：選択した水素原子と結合している原子を探す関数
"""
def search_nearest_atom_from_hydrogen(mol, a_index, a_search_distance=2.0) : 
    mol[a_index].symbol
    mol[a_index].position
    index_remain=-100 #残す側のindexを暫定的に-100番に
    output_norm=100 #出力するnorm(結合距離)を暫定的に100Åに

    for i in range(len(mol)):
        if mol[i].symbol=="H" : #水素の場合は除く
            continue
        search_vector = mol[a_index].position-mol[i].position     # axisからchangeへのvector取得
        search_norm=np.linalg.norm(search_vector)   #ノルム取得

        if 0 < search_norm < a_search_distance : 
            if search_norm < output_norm : #結合距離が小さいものを探せたら置き換える
                index_remain=i
                output_norm=search_norm
    
    if index_remain==-100 : #index_remainiが-100のときはindexが探せていないのでエラーに
        print(f"{'*'*30}\nindex_remain_error\n{'*'*30}")
    return index_remain
# def search_nearest_atom_from_hydrogen(mol, a_index) : 
#     return f_neighbor_atom(mol,a_index)[0]



"""くっつける側：指定したindexと距離が近い水素原子のindexを探す関数
"""
def search_nearest_hydrogen_index(atoms,a_index,a_thre_upper=2.0,a_thre_lower=0,a_all="F") : 
    atoms_distance = atoms.get_distances(a_index,range(len(atoms)), mic=True) # a_indexとの結合距離を全原子で取得
    min_dist_index = atoms_distance.argsort()[1]                              # 2番目に小さい距離を取得 (1番小さいのは自分自身なので除外)
    return [min_dist_index]

    # return [[i for i in f_neighbor_atom(atoms, a_index) if atoms.symbols[i]=="H"][0]]

    
def add_mole_verAddeditor(atoms_sbs,atoms_ads,indx_sbs_change,indx_ads_remain,a_bond_length=None,a_rotate=None, a_replace=True):
    try :
        # indx_sbs_remain=search_nearest_atom_from_hydrogen(atoms_sbs, a_index=indx_sbs_change, a_search_distance=2) #残す原子のindex取得
        # indx_ads_change=search_nearest_hydrogen_index(atoms_ads,a_index=indx_ads_remain)[0] #消す原子のindex取得
        
        indx_sbs_remain = f_neighbor_atom(atoms_sbs, indx_sbs_change)[0]                                             #残す原子のindexを取得
        indx_ads_change = [i for i in f_neighbor_atom(atoms_ads, indx_ads_remain) if atoms_ads.symbols[i]=="H"][0]   #消す原子のindexを取得
                
    except IndexError : 
        print(f"{'*'*30}\nIndex Error\n{'*'*30}")
    else :
        indx_sbs=[indx_sbs_remain,indx_sbs_change]
        indx_ads=[indx_ads_change,indx_ads_remain]
        if a_replace==True : 
            new_mol=add_mole_editor(atoms_sbs,atoms_ads,indx_sbs,indx_ads,a_bond_length=a_bond_length,a_rotate=a_rotate)
        else :
            new_mol=add_mole_editor(atoms_sbs,atoms_ads,indx_sbs,indx_ads,a_bond_length=a_bond_length,a_rotate=a_rotate, a_replace=False)
        return new_mol
    
    
    
"""Addeditor 水素挿入で使用する関数
"""
def insert_hydrogen_all_specify(mol,index_list, reverse = "F") : 
    target_index = index_list[0]
    nearest_atom_index = index_list[1]
    second_near_atom_index = index_list[2]
    mol_pos = mol.get_positions()

    #外積の取得
    vec01=mol_pos[nearest_atom_index]-mol_pos[target_index]
    vec12=mol_pos[second_near_atom_index]-mol_pos[target_index]
    vec_cross=np.cross(vec01,vec12)
    vec_cross_add=vec_cross/np.linalg.norm(vec_cross)*1.09
    if reverse=="T" : vec_cross_add = -vec_cross_add #reverse="T"だった場合は外積ベクトルを逆向き (左ねじの向き) にする

    #水素原子の座標を決定し、くっつける
    pos_add_hydrogen = mol_pos[target_index] + vec_cross_add
    hydrogen_atom = Atom(symbol="H",position=pos_add_hydrogen)
    mol.append(hydrogen_atom)
    
    return mol


def insert_hydrogen(mol,target_index,reverse="F") :
    target_index = target_index
    mol_pos = mol.get_positions()
    mol_all_distance = mol.get_all_distances()
    
    #insert_hydrogen_all_specify関数で使うindex取得
    mol_target_distance = mol_all_distance[target_index]
    mol_target_distance_argsorted = np.argsort(mol_target_distance) #argsortでは小さい順に並べたindexの配列が得られる
    nearest_atom_index = mol_target_distance_argsorted[1] #対象原子を除き1番目に距離が近い原子のindex取得
    second_near_atom_index = mol_target_distance_argsorted[2] #対象原子を除き2番目に距離が近い原子のindex取得
    index_list=[target_index, nearest_atom_index, second_near_atom_index]
    
    #関数使用
    insert_hydrogen_all_specify(mol,index_list,reverse=reverse)
    return mol












"""AddEditor
"""
class AddEditor:
    """Structure viewer/editor"""

    struct: List[Dict]  # structure used for nglview drawing.
    pot: float  # potential energy
    mforce: float  # mean average force
    show_force: bool = False
    show_axes: bool = False
    show_index: bool = False

   # def __init__(self, atoms: Atoms, w: int = 450, h: int = 450):
    def __init__(self, atoms=None, 
                 background_color = "#222222", 
                 w: int = 450, h: int = 470):
        """background_color : Default is "#222222"(=black). "White" can be used.
        """
        
        # 引数に何も指定されなかったらCH4を出力する。以下3行と__init__をatoms: Atomsにすれば元のスクリプトに戻る
        if atoms==None:
            atoms=smiles_to_ase_atoms("C")
            atoms.calc = ASECalculator(Estimator(calc_mode="CRYSTAL_U0", model_version="latest"))
        elif type(atoms)==str:
            atoms=smiles_to_ase_atoms(atoms)
            atoms.calc = ASECalculator(Estimator(calc_mode="CRYSTAL_U0", model_version="latest"))
        #####
        
        self.atoms = atoms
        self.undo_list = []
        self.redo_list = []
        # self.folder_name_addmole="./for_addmole"
        self.folder_name_addmole=os.path.dirname(__file__)+"/for_addmole"   # .pyファイルに入れるとき(__file__は実行中のファイルのパスを取得するが.pyじゃないとエラーに)
        #######################################################################
        
        
        """下のコメントアウトを外せばcalculatorの設定もできるようになるが紛らわしくなるのでコメントアウト
        """
#         #calculatorを引数で設定できるように
#         if (calc_mode==None) & (model_version==None) : 
#             pass
#         elif calc_mode == None : 
#             calc_mode = self.atoms.calc.estimator.calc_mode
#         elif model_version == None :
#             model_version = self.atoms.calc.estimator.model_version
#         else : 
#             pass 
            
#         if (calc_mode!=None) | (model_version!=None) :
#             print(f"Set Calculator : {calc_mode}, {model_version}")
#             estimator = Estimator(calc_mode=calc_mode, model_version=model_version) 
#             self.atoms.calc=ASECalculator(estimator)
        
        """ もともとのcalculatorを取得
        """ 
        try : 
            self.calc_mode = atoms.calc.estimator.calc_mode
            self.model_version = atoms.calc.estimator.model_version
        except : 
            msg = ("Atoms object has no calculator")
            raise ValueError(msg)
        else :
            self.estimator = Estimator(calc_mode=self.calc_mode, model_version=self.model_version)
            self.atoms.calc = ASECalculator(self.estimator)
            
        """ もともとのconstraintsを取得
        """
        try :                                                        # もともとのconstraintsを保存
            self.original_cons = self.atoms.constraints[0].index.copy()
        except :                                                     #  constがかかっていないとエラーになるので
            self.original_cons= []
        #######################################################################
        #self.atoms = atoms
        self.prev_atoms = []
        self.vh = NGLDisplay(atoms, w, h).gui
        self.v: NGLWidget = self.vh.view
        self.background_color = background_color
        
        ########################################
        # --- addmole attributes ---
        self.addmole_mol = "C"        
        self.addmol_view = nv.NGLWidget(width="400px", height="200px")
        self.addmol_view.background="lightgray"
        ########################################
        
        update_tooltip_atoms(self.v, self.atoms)
        self.recont()  # Add controller
        self.set_representation()
        self.set_atoms()
        self.pots = []
        self.traj = []

        self.cal_nnp()

        # --- force related attributes ---
        self.show_force = False
        self._force_components = []
        self.force_scale = 0.4
        self.force_color = [1, 0, 0]

        self.show_axes = False
        self._axes_components = []
        self.axes_scale = 0.4
        self.axes_color = [[1,0,0],[0,1,0],[0,0,1]]
        

        
    def clear_force(self):
        # Remove existing force components.
        for c in self._force_components:
            self.v.remove_component(c)  # Same with c.clear()
        self._force_components = []

    def clear_axes(self):
        # Remove existing axes components.
        for c in self._axes_components:
            self.v.remove_component(c)  # Same with c.clear()
        self._axes_components = []
        
    def add_force(self):
        try:
            c = add_force_shape(self.atoms, self.v, self.force_scale, self.force_color)
            self._force_components.append(c)
        except Exception as e:
            with self.vh.out_widget:
                print(traceback.format_exc(), file=sys.stderr)
            # `append_stderr` method shows same text twice somehow...
            # self.gui.out_widget.append_stderr(str(e))
            return
        
    def add_axes(self):
        try:
            c = add_axes_shape(self.atoms, self.v, self.axes_scale)
            self._axes_components.append(c)
        except Exception as e:
            with self.vh.out_widget:
                print(traceback.format_exc(), file=sys.stderr)
            # `append_stderr` method shows same text twice somehow...
            # self.gui.out_widget.append_stderr(str(e))
            return

    def display(self):
        #self.v.background =self.background_color
        display(self.vh)

    def recont(self):
        grid0 = GridspecLayout(2, 3, layout={"width":"460px"})
        self.vh.xplusview  = Button(description="View X+")
        self.vh.yplusview  = Button(description="View Y+")
        self.vh.zplusview  = Button(description="View Z+")
        self.vh.xminusview = Button(description="View X-")
        self.vh.yminusview = Button(description="View Y-")
        self.vh.zminusview = Button(description="View Z-")
        self.vh.xplusview.on_click(self.rotate_view)
        self.vh.yplusview.on_click(self.rotate_view)
        self.vh.zplusview.on_click(self.rotate_view)
        self.vh.xminusview.on_click(self.rotate_view)
        self.vh.yminusview.on_click(self.rotate_view)
        self.vh.zminusview.on_click(self.rotate_view)
        grid0[0, 0] = self.vh.xplusview
        grid0[0, 1] = self.vh.yplusview
        grid0[0, 2] = self.vh.zplusview
        grid0[1, 0] = self.vh.xminusview
        grid0[1, 1] = self.vh.yminusview
        grid0[1, 2] = self.vh.zminusview

#         self.vh.setatoms = FloatSlider(
#             min=-50, max=50, step=0.1, value=50, 
#             description="atoms z>"
#         )
#        self.vh.setatoms.observe(self.set_atoms)

        self.vh.selected_atoms_label = Label("Selected atoms:")
        self.vh.selected_atoms_textarea = Textarea(layout={"width": "295px"})
        selected_atoms_hbox = HBox(
            [self.vh.selected_atoms_label, self.vh.selected_atoms_textarea]
        )
        grid1 = GridspecLayout(1, 3)
        self.vh.delete = Button(description="Delete")
        self.vh.replace = Button(description="Replace to")
        self.vh.replace_symbol = Text(value="H", layout={"width": "50px"})
        self.vh.delete.on_click(self.delete)
        self.vh.replace.on_click(self.replace)
        grid1[0, 0] = self.vh.delete
        grid1[0, 1] = self.vh.replace
        grid1[0, 2] = self.vh.replace_symbol

##################################################################################  
        """Setatoms per x,y,z axis
        """
        self.vh.setatoms_x = FloatRangeSlider(min=-50, max=50, step=0.1, value=[-50,50], layout={"width":"375px"}, description="atoms x:")
        self.vh.setatoms_x.observe(self.set_atoms)
        self.vh.setatoms_y = FloatRangeSlider(min=-50, max=50, step=0.1, value=[-50,50], layout={"width":"375px"}, description="atoms y:")
        self.vh.setatoms_y.observe(self.set_atoms)
        self.vh.setatoms_z = FloatRangeSlider(min=-50, max=50, step=0.1, value=[0,50], layout={"width":"375px"}, description="atoms z:")
        self.vh.setatoms_z.observe(self.set_atoms)
        
        """Display in "selected atoms" only selected elements
        """
        self.vh.display_only_selected_elements = RadioButtons(options=['All', 'Only selected elements'], value = "All", disabled=False)
        
        self.vh.selected_elements_textarea = Textarea(value="H, C")
        
        """Undo and Redo for Add_mole function 
        """
        self.vh.undo = Button(description="Undo ↰", layout={"width":"90px"})
        self.vh.undo.on_click(self.undo)
        self.vh.redo = Button(description="Redo ↱", layout={"width":"90px"})
        self.vh.redo.on_click(self.redo)
        
        """ Clear Textarea
        """
        self.vh.cleartextarea = Button(description="Clear", layout={"width":"100px"})
        self.vh.cleartextarea.on_click(self.clear_textarea)
        
        """Set calculator
        """
        grid10 = GridspecLayout(2, 1)
        self.vh.setcalc_calculator = Dropdown(options=['CRYSTAL', 'CRYSTAL_PLUS_D3', 'CRYSTAL_U0', 'CRYSTAL_U0_PLUS_D3', 'MOLECULE'], description = "calculator", value="CRYSTAL_U0")
        self.vh.setcalc_version    = Dropdown(options=["latest", "v0.0.0", "v1.0.0", "v1.1.0", "v2.0.0", "v3.0.0", "v4.0.0","v5.0.0","v6.0.0","v7.0.0"], description = "version", value="latest")
        self.vh.setcalc_button     = Button(description="set", layout={"width":"100px"})
        self.vh.setcalc_button.on_click(self.setcalc)
        grid10[0, 0] = self.vh.setcalc_calculator
        grid10[1, 0] = HBox([self.vh.setcalc_version, self.vh.setcalc_button])
        
        """Exchange the first and second indices in 'Selected atoms'
        """
        self.vh.exchange_index = Button(description="Exchange", layout={"width":"100px"})
        self.vh.exchange_index.on_click(self.exchange_index)
        grid_exchange_index = HBox([self.vh.exchange_index])
        
        """Constraints
        """
        self.vh.setconstraints        = Button(description="Set constraints", layout={"width":"150px"})
        self.vh.delconstraints        = Button(description="Del constraints", layout={"width":"150px"})
        self.vh.setconstraints.on_click(self.setconstraints)
        self.vh.delconstraints.on_click(self.delconstraints)
        grid_constraints = HBox([self.vh.setconstraints, self.vh.delconstraints])
        
        """Insert molecules as smiles
        """
        self.vh.insertMol_button    = Button(description="Insert Molecule", layout={"width":"405px"})
        self.vh.insertMol_direction = Dropdown(options=['x+', 'x-', 'y+', 'y-', 'z+', 'z-'], description = "Direction", value="z+", layout={"width":"150px"})
        self.vh.insertMol_position  = FloatSlider(min=0, max=10, step=0.1, value=1, description="Position", layout={"width":"275px"})
        self.vh.insertMol_smiles    = Text(value="H", layout={"width": "405px"}, description="Smiles")
        self.vh.insertMol_button.on_click(self.insert_molecule)
        
        grid_insertMol = VBox([self.vh.insertMol_smiles,
                              HBox([self.vh.insertMol_direction,
                                    self.vh.insertMol_position]),
                              self.vh.insertMol_button])
        
        """ Add Valence
        """
        self.vh.addvalence = Button(description="Add Valence", layout={"width":"200px"})
        self.vh.addvalence.on_click(self.addvalence)
        self.vh.miniopt_checkbox = Checkbox(value=True,description="Mini opt")
        
        self.vh.addvalence_symbol = Text(value="H", layout={"width": "50px"})
        
        """ Add Functional Group
        """
        self.vh.addfunctional = Button(description="Set", layout={"width":"100px"})
        self.vh.addfunctional.on_click(self.add_functionalgroup)
        
        self.vh.addfunctional_smiles = Text(value="c1ccccc1", layout={"width": "350px"}, description="Smiles") # 405px
        
        """ Textbox of Opt
        """
        self.vh.optlog_textarea = Textarea(layout={"width":"450px", "height":"155px"}, disabled=True)        
        self.vh.optlog_textarea.value = "This is the output place of Opt."

        
        """ 周期表のボタン
        """
        grid_element = GridspecLayout(3, 8, layout={"width":"405px"})
        words = [["H",                                    "He"], 
                 ["Li", "Be", "B",  "C",  "N", "O", "F" , "Ne"],
                 ["Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar"]
                ]
        # 第1周期
        grid_element[0,0] = Button(description=words[0][0], layout={"width":"46px"},value="H")
        grid_element[0,7] = Button(description=words[0][1], layout={"width":"46px"},value="He")
        
        grid_element[0,0].on_click(self.set_atoms_addmole)
        grid_element[0,7].on_click(self.set_atoms_addmole)
        
        # 第2周期
        for i_group in range(len(words[1])):
            grid_element[1,i_group] = Button(description=words[1][i_group], layout={"width":"46px"},value=words[1][i_group])
            grid_element[1,i_group].on_click(self.set_atoms_addmole)
        # 第3周期
        for i_group in range(len(words[2])):
            grid_element[2,i_group] = Button(description=words[2][i_group], layout={"width":"46px"},value=words[2][i_group])
            grid_element[2,i_group].on_click(self.set_atoms_addmole)
        
        """ Bond Typeのimgを出力するボタン
        """
        img_file = open(f"{self.folder_name_addmole}/AddMole_image_all.jpg", "rb")
        add_image = img_file.read()
        add_image_show = Image(value=add_image, format='jpg', width=400, height=400)

        """ 結合形式のボタン
        """
        self.vh.add_image_buttons = Dropdown(options=['1-coordination', '2-coordination', '3-coordination', '4-coordination'], 
                                             description='', value='4-coordination', disabled=False, layout={"width":"200px"})
        self.vh.add_image_buttons.observe(self.on_dropdown_add_image_buttons_change, names='value')

        # self.vh.addmole_button = Button(description="Add Molecule", layout={"width":"180px"})   # もとは400px
        # self.vh.addmole_button.on_click(self.addmole)
        
        """ addmoleをjslinkで対応させて1つにまとめる(adm_flagments)
        """
        addmole_flagment = VBox([HTML(value =f"<b><font size='2'> ■ Select Element </font><b>"),
                                grid_element, 
                                HBox([HTML(value =f"<b><font size='2'> ■ Select Bond Type {'&nbsp;'*20} </font><b>"), self.vh.add_image_buttons]), 
                                add_image_show, 
                                self.addmol_view])
        addmole_functional_flagment = VBox([HBox([self.vh.addfunctional_smiles,
                                                  self.vh.addfunctional]),
                                            self.addmol_view])
        adm_stack = widgets.Stack([addmole_flagment, addmole_functional_flagment], selected_index=0)
        adm_dropdown = Dropdown(options=["Element","Functional Group"])
        widgets.jslink((adm_dropdown, 'index'), (adm_stack, 'selected_index'))
        adm_flagments = VBox([HBox([adm_dropdown, self.vh.miniopt_checkbox]), 
                              adm_stack])
        
        """DeleteReplace、Add mole、Insert moleculeをAccordionでまとめる
        """
        self.vh.accordion_add = Accordion(children=[grid1,
                                            adm_flagments,
                                            VBox([HTML(value =f"<font size='2'>  {'&nbsp;'*2} #Any atom can be added to the vacant space for Selected atoms.</font>"),
                                                  HBox([self.vh.addvalence_symbol, self.vh.miniopt_checkbox, self.vh.addvalence])]),
                                            VBox([HTML(value =f"<font size='2'>  {'&nbsp;'*2} #Any molecule can be added in the specified direction for Selected atoms. </font>"),grid_insertMol]),
                                           ])
        self.vh.accordion_add.set_title(0, '0. Delete or Replace Atoms')
        self.vh.accordion_add.set_title(1, '1. Add Molecule')
        self.vh.accordion_add.set_title(2, '2. Add Valence')
        self.vh.accordion_add.set_title(3, '3. Insert Molecule')
        

        
        self.vh.accordion_add.selected_index=None # 全てのセクションを閉じた状態に設定
        self.vh.accordion_add.observe(self.on_accordion_change, names="selected_index")
        
        
        self.vh.touch_mode_select = Dropdown(options=['Normal','Add molecule','Add Valence','Delete'], description = "Clicked mode", value="Normal", layout={"width":"350px"}, disabled=True)
        
        
        """ addmoleの初期条件で計算
        """
        self.f_addmole_str_to_atoms()   # addmoleの初期条件("C","4-coordinate")でaddmoleのatoms作成。本当はもっと上にもっていきたいがadd_imajge_buttonsの下じゃないといけないのでここに置いてる。
        self.addmol_component = self.addmol_view.add_component(nv.ASEStructure(self.atoms_base_addmole))   # addmol_viewにAtomsを追加
        self.index_remain_adv = 0 # 残す原子のindex
        self.index_h_adv      = [i for i in f_neighbor_atom(self.atoms_base_addmole, self.index_remain_adv) if self.atoms_base_addmole.symbols[i]=="H"][0] # 残す原子に結合しているH原子のindex

        # addmoleの消す水素原子のindexをオレンジ色に表示
        selection_str=f'@{self.index_h_adv}'
        self.addmol_view.add_representation("ball+stick",selection=selection_str,color="darkorange",aspectRatio=4)
        
        # --- Register callback ---
        self.addmol_view.observe(self._on_picked_change_addmolview, names=["picked"])

        ##################################################################################
        
        self.vh.move = FloatSlider(
            min=0.1, max=2, step=0.1, value=0.5, description="", layout={"width":"250px"}
        )

        grid2 = GridspecLayout(2, 3, layout={"width":"460px"})
        self.vh.xplus = Button(description="X+")
        self.vh.xminus = Button(description="X-")
        self.vh.yplus = Button(description="Y+")
        self.vh.yminus = Button(description="Y-")
        self.vh.zplus = Button(description="Z+")
        self.vh.zminus = Button(description="Z-")
        self.vh.xplus.on_click(self.move)
        self.vh.xminus.on_click(self.move)
        self.vh.yplus.on_click(self.move)
        self.vh.yminus.on_click(self.move)
        self.vh.zplus.on_click(self.move)
        self.vh.zminus.on_click(self.move)
        grid2[0, 0] = self.vh.xplus
        grid2[0, 1] = self.vh.yplus
        grid2[0, 2] = self.vh.zplus
        grid2[1, 0] = self.vh.xminus
        grid2[1, 1] = self.vh.yminus
        grid2[1, 2] = self.vh.zminus

        self.vh.rotate = FloatSlider(
            min=1, max=90, step=1, value=30, description="", layout={"width":"250px"}
        )
        grid3 = GridspecLayout(2, 3, layout={"width":"460px"})
        self.vh.xplus2 = Button(description="X+")
        self.vh.xminus2 = Button(description="X-")
        self.vh.yplus2 = Button(description="Y+")
        self.vh.yminus2 = Button(description="Y-")
        self.vh.zplus2 = Button(description="Z+")
        self.vh.zminus2 = Button(description="Z-")
        self.vh.xplus2.on_click(self.rotate)
        self.vh.xminus2.on_click(self.rotate)
        self.vh.yplus2.on_click(self.rotate)
        self.vh.yminus2.on_click(self.rotate)
        self.vh.zplus2.on_click(self.rotate)
        self.vh.zminus2.on_click(self.rotate)
        grid3[0, 0] = self.vh.xplus2
        grid3[0, 1] = self.vh.yplus2
        grid3[0, 2] = self.vh.zplus2
        grid3[1, 0] = self.vh.xminus2
        grid3[1, 1] = self.vh.yminus2
        grid3[1, 2] = self.vh.zminus2
        
        #########################################################
        """指定したベクトルv_01を軸として回転
        """
        self.vh.rotate_specified_vec_p = Button(description="Rotate +")
        self.vh.rotate_specified_vec_p.on_click(self.rotate_specified_vec)
        self.vh.rotate_specified_vec_m = Button(description="Rotate -")
        self.vh.rotate_specified_vec_m.on_click(self.rotate_specified_vec)
        self.vh.specified_vec0 = BoundedIntText(value="0", max=10000, layout={"width": "60px"})
        self.vh.specified_vec1 = BoundedIntText(value="0", max=10000, layout={"width": "60px"})
        
        
        """指定したベクトルv_01を軸として平行移動
        """
        self.vh.move_specified_plus  = Button(description="Move +")
        self.vh.move_specified_minus = Button(description="Move -")
        self.vh.move_specified_plus.on_click(self.move_specified)
        self.vh.move_specified_minus.on_click(self.move_specified)
        #########################################################

        self.vh.nnptext = Textarea(disabled=True,layout={"width": "210px", "height":"33px"}) #layout=Layout(height="50px")にすると3行分全部見える

        # self.vh.opt_step = IntSlider(
        #     min=0,
        #     max=100,
        #     step=1,
        #     value=10,
        #     description="Opt steps",
        # )
        self.vh.opt_step = BoundedIntText(min=0, max=10000, step=50,   value=1000,  description="steps", layout={"width":"145px"})
        self.vh.opt_fmax = BoundedFloatText(min=0, max=10000, step=0.001, value=0.001, description="fmax ", layout={"width":"145px"})
        self.vh.opt_maxstep = BoundedFloatText(min=0, max=10000, step=0.01, value=0.2, description="maxstep ", layout={"width":"145px"})
        
        # self.vh.based_constraint_checkbox = Checkbox(value=True, description="Opt based on constraints")
        self.vh.constraint_checkbox = Checkbox(value=False,description="Opt only selected atoms")
        # self.vh.optlog_checkbox = Checkbox(value=False,description="Show logfile")
        self.vh.opt_algorithm_dropdown = Dropdown(options=['FIRELBFGS','LBFGS','FIRE'], description = "Algorithm", value="FIRELBFGS", layout={"width":"304px"}) # layoutはstep,fmaxの幅との調整
        self.vh.run_opt_button = Button(
            description="Run opt",
            tooltip="Execute FIRELBFGS optimization.",
        )
        self.vh.run_opt_button.on_click(self.run_opt)
        opt_hbox = HBox([self.vh.constraint_checkbox, self.vh.run_opt_button])

        self.vh.filename_text = Text(value="screenshot.png", description="filename: ")
        self.vh.download_image_button = Button(
            description="download image",
            tooltip="Download current frame to your local PC",
        )
        self.vh.download_image_button.on_click(self.download_image)
        ###################################################################################################
        self.vh.download_image_button_transparent = Button(
            description="download image (transparent)",
            tooltip="Download current frame to your local PC with transparent background",
            layout={"width":"300px"}
        )
        self.vh.download_image_button_transparent.on_click(self.download_transparent_image)
        #####################################################################################################
        self.vh.save_image_button = Button(
            description="save image",
            tooltip="Save current frame to file.\n"
            "Currently .png and .html are supported.\n"
            "It takes a bit time, please be patient.",
        )
        self.vh.save_image_button.on_click(self.save_image)
            
        self.vh.show_force_checkbox = Checkbox(
            value=self.show_force,
            description="Show force",
            indent=True, 
            layout={'width': '200px'},
        )
        self.vh.show_force_checkbox.observe(self.show_force_event)
        
        self.vh.show_axes_checkbox = Checkbox(
            value=self.show_axes,
            description="Show axes",
            indent=True, 
            layout={'width': '200px'},
        )
        self.vh.show_axes_checkbox.observe(self.show_axes_event)
        
        ########################################################################################
        """show indexができるように
        """
        self.vh.show_index_checkbox = Checkbox(
            value=self.show_index,
            description="Show index",
            indent=False, 
            layout={'width': '200px'},
        )
        self.vh.show_index_checkbox.observe(self.show_index_event)
        
        """選択した原子だけshow index
        """
        self.vh.show_index_one_part_checkbox = Checkbox(
            #value=self.show_index,
            description="Show index in 'Selected atoms'",
            indent=False, 
            layout={'width': '200px'},
        )
        self.vh.show_index_one_part_checkbox.observe(self.show_index_event)
    
        
        grid_cb = GridspecLayout(2, 2)
        grid_cb[0, 0] = self.vh.show_force_checkbox
        grid_cb[0, 1] = self.vh.show_index_checkbox
        grid_cb[1, 0] = self.vh.show_axes_checkbox
        grid_cb[1, 1] = self.vh.show_index_one_part_checkbox
        ########################################################################################
        
        
        self.vh.out_widget = Output(layout={"border": "0px solid black"})

        self.vh.update_display = Button(
            description="update_display",
            tooltip="Refresh display. It can be used when target atoms is updated in another cell..",
        )
        self.vh.update_display.on_click(self.update_display)

        ###################################
        """Otherの部分をAccordionでまとめる
        """
        self.vh.accordion_other = Accordion(children=[VBox([HTML(value =f"<font size='2'>  #Exchange the first and second indices in 'Selected atoms' &nbsp; </font>"),
                                                            grid_exchange_index]),
                                                      VBox([HTML(value =f"<font size='2'>  #Set constraints in 'Selected atoms' or Delete all constraints. &nbsp; </font>"),
                                                      grid_constraints]),
                                                      VBox([self.vh.filename_text,
                                                            HBox([self.vh.download_image_button, self.vh.save_image_button]),
                                                            self.vh.download_image_button_transparent]),
                                                      grid10
                                                     ])
        self.vh.accordion_other.set_title(0, '0. Exchange indices')
        self.vh.accordion_other.set_title(1, '1. Constraints')
        self.vh.accordion_other.set_title(2, '2. Save Image')
        self.vh.accordion_other.set_title(3, '3. Set Calculator')
        

        
        self.vh.accordion_other.selected_index=None # 全てのセクションを閉じた状態に設定
        
        
        
        ###################################
        
################################################################################
        """tab化スクリプト
        """
        bullet_points="■"
        
        original_controlbox = [i for i in list(self.vh.control_box.children)] #もともとのcontrol_boxをリスト内包表記でほどいた
    
        
        viewer = VBox(
            [HTML(value =f"<b><font size='4'> {bullet_points} Details </font><b>")]
            + original_controlbox
            +[
            HTML(value =f"<b><font size='4'> {bullet_points} ViewPoint Change </font><b>"),
            grid0,
            #HBox([self.vh.show_force_checkbox, self.vh.update_display]),  
            # HBox([self.vh.show_axes_checkbox]),
            grid_cb,
            self.vh.out_widget
        ])
        
        selection_atoms = VBox([
            HTML(value =f"<b><font size='4'> {bullet_points} Selection of Range in 'Selected atoms'</font><b>"),
            HBox([HTML(value ="<font size='2'> &nbsp; &nbsp; </font>"), self.vh.setatoms_x]),
            HBox([HTML(value ="<font size='2'> &nbsp; &nbsp; </font>"), self.vh.setatoms_y]),
            HBox([HTML(value ="<font size='2'> &nbsp; &nbsp; </font>"), self.vh.setatoms_z]),
            HTML(value =f"<b><font size='4'> {bullet_points} Selection of Elements in 'Selected atoms'</font><b>"),
            self.vh.display_only_selected_elements,
            self.vh.selected_elements_textarea
        ])
        
        editor = VBox([
            HBox([HTML(value =f"<b><font size='4'> {bullet_points} Move &nbsp; </font><b>"), self.vh.move, HTML(value =f"<font size='2'> Å </font>")]),
            grid2,
            HBox([HTML(value =f"<b><font size='4'> {bullet_points} Rotate </font><b>"), self.vh.rotate, HTML(value =f"<font size='2'> degree </font>")]),
            grid3,
            HBox([HTML(value = f"<b><font size='4'> {bullet_points} Move or Rotate on the axis of Vec</font><b>"),
                  self.vh.specified_vec0,
                  self.vh.specified_vec1
                 ]),            
            HBox([self.vh.move_specified_plus,  self.vh.rotate_specified_vec_p]),
            HBox([self.vh.move_specified_minus, self.vh.rotate_specified_vec_m]),
            self.vh.out_widget
        ])
        
        addition = VBox([
            self.vh.touch_mode_select,
            self.vh.accordion_add,
            self.vh.out_widget
        ])
        
        mini_opt_box = VBox([
            HTML(value =f"<b><font size='4'> {bullet_points} Opt </font><b>"),
            HBox([self.vh.opt_step,self.vh.opt_fmax,self.vh.opt_maxstep]),
            self.vh.opt_algorithm_dropdown,
            opt_hbox,
            self.vh.optlog_textarea,
            self.vh.out_widget
        ])
        
        other = VBox([self.vh.accordion_other,
                      self.vh.out_widget
        ])
        
        tab_children = [viewer, selection_atoms, editor, addition, mini_opt_box, other]
        
        tab_titles = {0:'Viewer', 1:'Range', 2:'Editing', 3:'Addition', 4:'Opt', 5:'Other'} # ipywidgets<=7.7.1用
        self.tab = Tab(children=tab_children, _titles=tab_titles, layout={"width":"505px"})      # ノートPCにちょうど収まるサイズは465px
        self.tab.titles=['Viewer', 'Range', 'Editing', 'Addition', 'Opt', 'Other']               # ipywidgets>=8.1.3用のタイトル。
        self.tab.observe(self.on_tab_change, names="selected_index")                                  # 開いているタブを変更するとon_tab_changeが実行される
################################################################################
        
#        r = list(self.vh.control_box.children) #viewerに統合した
        
        """空白の位置を調整するためにHBoxの中に特殊文字 "&nbsp" を用いた
        """
        r = [HBox([HTML(value ="<font size='2'> &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; NNP text: </font>"), self.vh.nnptext, self.vh.undo, self.vh.redo]),
             HBox([HTML(value ="<font size='2'> &nbsp; &nbsp; </font>"), self.vh.setatoms_z]),
             HBox([HTML(value ="<font size='2'> &nbsp; </font>"), selected_atoms_hbox, self.vh.cleartextarea])
            ]
        r += [self.tab]

        self.vh.control_box.children = tuple(r)
        # self.vh.view = [self.vh.view, HBox([self.vh.undo, self.vh.redo])]
        
        # --- Register callback ---
        self.v.observe(self._on_picked_changed_set_atoms, names=["picked"])

    def set_representation(self, bcolor: str = "white", unitcell: bool = False):
        #self.v.background = bcolor
        self.v.background = self.background_color
        self.struct = get_struct(self.atoms)
        self.v.add_representation(repr_type="ball+stick")
        if unitcell:
            self.v.add_representation(repr_type="unitcell")
        self.v.control.spin([0, 1, 0], pi * 1.1)
        self.v.control.spin([1, 0, 0], -pi * 0.45)
        # It's necessary to update indices of atoms specified by `get_struct` method.
        self.v._remote_call("replaceStructure", target="Widget", args=self.struct)

    # def get_struct(self, atoms: Atoms, ext="pdb") -> List[Dict]:
    #     # For backward compatibility...
    #     return get_struct(atoms, ext=ext)

    def cal_nnp(self):
        mforce = (((self.atoms.get_forces()) ** 2).sum(axis=1).max()) ** 0.5
        pot = (
            self.atoms.get_potential_energy()
        )  # Faster to calculate energy after force.
        self.pot = pot
        self.mforce = mforce
        self.vh.nnptext.value = f"pot energy : {pot:.2f} eV\nmax force  : {mforce:.4f} eV/A\ncalculator : {self.model_version} \n {' '*11} {self.calc_mode}"
        self.pots += [pot]
        self.traj += [self.atoms.copy()]

    def _update_Q(self):
        # Update `var atoms_pos` inside javascript.
        atoms = self.atoms
        if atoms.get_pbc().any():
            _, Q = atoms.cell.standard_form()
        else:
            Q = np.eye(3)
        Q_str = str(Q.tolist())
        var_str = f"this._Q = {Q_str}"
        self.v._execute_js_code(var_str)

    def _update_structure(self):
        struct = get_struct(self.atoms)
        self.struct = struct
        self.v._remote_call("replaceStructure", target="Widget", args=struct)
        self._update_Q()

    def update_display(self, clicked_button: Optional[Button] = None):
        # Force must be cleared before updating structure...
        self.clear_force()
        self.clear_axes()
        self._update_structure()
        self.cal_nnp()
        if self.show_force:
            self.add_force()
        if self.show_axes:
            self.add_axes()

# SurfaceEditorの機能 : 指定したz軸でindex取得 (改良したのでここは不要)
#     def set_atoms(self, slider: Optional[FloatSlider] = None):
#         """Update text area based on the atoms position `z` greater than specified value."""
#         smols = [
#             i for i, atom in enumerate(self.atoms) if atom.z >= self.vh.setatoms.value
#         ]
#         self.vh.selected_atoms_textarea.value = ", ".join(map(str, smols))
        
################################################################################################
    def set_atoms(self, slider: Optional[FloatSlider] = None):
        """Allows specifying on each of the x, y, and z axes.
           Only selected elements がTrueだった場合はその元素だけindexを取得できるようにする"""
        
        # Only selected elemtnts = Flaseの場合は指定したxyzの範囲内の全原子のindexを取得
        if self.vh.display_only_selected_elements.value!='Only selected elements': 
            smols = [
                i for i, atom in enumerate(self.atoms) \
                if (self.vh.setatoms_x.value[0] <= atom.x <= self.vh.setatoms_x.value[1]) &\
                   (self.vh.setatoms_y.value[0] <= atom.y <= self.vh.setatoms_y.value[1]) &\
                   (self.vh.setatoms_z.value[0] <= atom.z <= self.vh.setatoms_z.value[1])
            ] #FloatRangeSliderはtupleかlistかは分からないけど[min, max]の形で与えられるため最小値は[0]で、最大値は[1]で取得
        
        # Only selected elements = Trueの場合は指定したxyzの範囲内かつ指定した元素のindexのみを取得
        else :
            try:
                selected_elements = self.vh.selected_elements_textarea.value.split(",")
                selected_elements_list = [
                    str(a.strip()) for a in selected_elements if a.strip() != ""
                ]
                
                atoms_chemical_symbols = self.atoms.get_chemical_symbols()
                
            except : print("None in selected elements")
            
            smols = [
                i for i, atom in enumerate(self.atoms) \
                if (self.vh.setatoms_x.value[0] <= atom.x <= self.vh.setatoms_x.value[1]) &\
                   (self.vh.setatoms_y.value[0] <= atom.y <= self.vh.setatoms_y.value[1]) &\
                   (self.vh.setatoms_z.value[0] <= atom.z <= self.vh.setatoms_z.value[1]) &\
                   (atoms_chemical_symbols[i] in selected_elements_list)
            ] #FloatRangeSliderはtupleかlistかは分からないけど[min, max]の形で与えられるため最小値は[0]で、最大値は[1]で取得            
            
        
        self.vh.selected_atoms_textarea.value = ", ".join(map(str, smols))
    
        self.show_one_part_index() #show index in Selected atomsにチェックボックスが入っていた場合のみSelected atoms内の原子のlabelを追加
        
################################################################################################

    def get_selected_atom_indices(self) -> List[int]:
        try:
            selected_atom_indices = self.vh.selected_atoms_textarea.value.split(",")
            selected_atom_indices = [
                int(a.strip()) for a in selected_atom_indices if a.strip() != ""
            ]
            return selected_atom_indices
        except Exception as e:
            with self.vh.out_widget:
                print(traceback.format_exc(), file=sys.stderr)
            # `append_stderr` method shows same text twice somehow...
            # self.gui.out_widget.append_stderr(str(e))
        return []

    def _on_picked_changed_set_atoms(self, change: Bunch):
        # print(type(change), change)  # It has "name", "old", "new" keys.
        selected_atom_indices = self.get_selected_atom_indices()
        # Ex. picked format
        # {'atom1': {'index': 15, 'residueIndex': 0, 'resname': 'MOL', ... 'name': '[MOL]1:A.FE'}, 'component': 0}
        index: int = self.v.picked.get("atom1", {}).get("index", -1)
        # print(index)
        # print(self.v.picked)
        if index != -1:
            selected_atom_indices.append(index)
        # else:
        #     print(f"[ERROR] Unexpected format: v.picked {self.v.picked}")   # 頻繁に出てくるので消した
        selected_atom_indices = list(sorted(set(selected_atom_indices)))
        self.vh.selected_atoms_textarea.value = ", ".join(
            map(str, selected_atom_indices)
        )
        
        ##########
        # touch_modeによって原子をクリックしたときの操作を変える
        if index!=-1 and self.vh.touch_mode_select.value=='Add molecule':
            self.f_addmole(added_index=index)
        elif index!=-1 and self.vh.touch_mode_select.value=='Add molecule(Functional)':
            self.f_add_functionalgroup(added_index=index) 
        elif index!=-1 and self.vh.touch_mode_select.value=='Add Valence':
            self.append_tmpatoms_to_atomslist(self.atoms)
            added_index=index
            
            bond_length = get_covalent_radii(self.vh.addvalence_symbol.value, self.atoms.symbols[added_index])   # 新しく作る結合の推定共有結合距離の取得
            self.atoms.append(Atoms(self.vh.addvalence_symbol.value, positions=[get_far_position(self.atoms, added_index, distance=bond_length)])[0])   # 距離が離れているときに原子追加

            self.estimator  = Estimator(calc_mode=self.calc_mode, model_version=self.model_version)
            self.atoms.calc = ASECalculator(self.estimator)      

            if self.vh.miniopt_checkbox.value:   # checkbox==Trueだったら少しだけopt
                opt = FIRELBFGS(self.atoms, maxstep_fire=0.5, maxstep_lbfgs=0.5, logfile=None)
                opt.run(fmax=0.5,steps=5)
                
            self.update_display()
                
        elif index!=-1 and self.vh.touch_mode_select.value=='Delete':
            self.append_tmpatoms_to_atomslist(self.atoms)
            del self.atoms[index]
                
            self.update_display()
            

        # print(self.vh.accordion_add.selected_index)   #開いているaccordionがどれかを出力
        
        ##########
        
        self.show_one_part_index() #show index in Selected atomsにチェックボックスが入っていた場合のみSelected atoms内の原子のlabelを追加

        
    def _on_picked_change_addmolview(self, change: Bunch):
        selected_atom_indices = self.get_selected_atom_indices()
        # Ex. picked format
        # {'atom1': {'index': 15, 'residueIndex': 0, 'resname': 'MOL', ... 'name': '[MOL]1:A.FE'}, 'component': 0}
        index: int = self.addmol_view.picked.get("atom1", {}).get("index", -1)
        # print(index)
        # print(self.v.picked)
        if (index != -1) and (self.atoms_base_addmole.symbols[index]=="H"):
            self.index_h_adv      = index
            self.index_remain_adv = [i for i in f_neighbor_atom(self.atoms_base_addmole, self.index_h_adv)][0]

            # ---addmolのviewで表示するAtoms更新 ---#
            self.addmol_view.remove_component(self.addmol_component)                                           # addmol_viewに追加したAtomsを削除
            self.addmol_component = self.addmol_view.add_component(nv.ASEStructure(self.atoms_base_addmole))   # addmol_viewに追加するAtoms

            # addmoleの消す水素原子のindexをオレンジ色に表示
            selection_str=f'@{self.index_h_adv}'
            self.addmol_view.add_representation("ball+stick",selection=selection_str,color="darkorange",aspectRatio=4)
        
        
    def rotate_view(self, clicked_button: Button):
        if clicked_button is self.vh.xplusview:
            self.v.control.rotate([-0.5, 0.5, 0.5, 0.5])
        elif clicked_button is self.vh.yplusview:
            self.v.control.rotate([-1 / sqrt(2), 0, 0, 1 / sqrt(2)])
        elif clicked_button is self.vh.zplusview:
            self.v.control.rotate([0, 1, 0, 0])
        elif clicked_button is self.vh.xminusview:
            self.v.control.rotate([-0.5, -0.5, -0.5, 0.5])
        elif clicked_button is self.vh.yminusview:
            self.v.control.rotate([0, 1 / sqrt(2), 1 / sqrt(2), 0])
        elif clicked_button is self.vh.zminusview:
            self.v.control.rotate([0, 0, 0, 1])
        else:
            raise ValueError("Unexpected button", clicked_button.description)

    def delete(self, clicked_button: Button):
        self.append_tmpatoms_to_atomslist(self.atoms)
        atom_indices = self.get_selected_atom_indices()
        atom_indices.sort(reverse=True)
        del self.atoms[atom_indices]
        self.update_display()

    def replace(self, clicked_button: Button):
        self.append_tmpatoms_to_atomslist(self.atoms)
        atom_indices = self.get_selected_atom_indices()
        replace_symbol = self.vh.replace_symbol.value
        try:
            self.atoms.symbols[atom_indices] = replace_symbol
        except:
            pass
        self.update_display()

##########################################################################

    def append_tmpatoms_to_atomslist(self, a_atoms):
        """ 現在のatoms型をatoms_listに保存する関数(ボタン関係なし、undo・redoボタンで使用)
        """
        self.undo_list.append(a_atoms.copy())
        self.estimator = Estimator(calc_mode=self.calc_mode, model_version=self.model_version)
        self.undo_list[-1].calc = ASECalculator(self.estimator)
        self.redo_list = []      
        
    def set_atoms_addmole(self, clicked_button: Button):
        """ addmole_molを周期表を押したときの元素で更新
        """
        self.addmole_mol = clicked_button.description
        self.f_addmole_str_to_atoms()
        self.index_remain_adv = 0 # atoms_base_addmoleの残す原子の初期index
        self.index_h_adv      = [i for i in f_neighbor_atom(self.atoms_base_addmole, self.index_remain_adv) if self.atoms_base_addmole.symbols[i]=="H"][0] # 残す原子に結合しているH原子のindex

        
        # ---addmolのviewで表示するAtoms更新 ---#
        self.addmol_view.remove_component(self.addmol_component)                                           # addmol_viewに追加したAtomsを削除
        self.addmol_component = self.addmol_view.add_component(nv.ASEStructure(self.atoms_base_addmole))   # addmol_viewに追加するAtoms
        
        # addmoleの消す水素原子のindexをオレンジ色に表示
        selection_str=f'@{[i for i in f_neighbor_atom(self.atoms_base_addmole, 0) if self.atoms_base_addmole.symbols[i]=="H"][0]}'
        self.addmol_view.add_representation("ball+stick",selection=selection_str,color="darkorange",aspectRatio=4)
        
    # def addmole(self, clicked_button: Button):
    #     """ ボタンを押したときf_addmol関数を呼ぶ
    #     """
    #     added_index = int(self.get_selected_atom_indices()[0]) #Selected atoms内の1番最初のindexだけ取得
    #     self.f_addmole(added_index=added_index)   
        
    def add_functionalgroup(self, clicked_button: Button):
        """ ボタンを押したらself.atoms_base_addmoleを指定したsmilesの分子に更新しviewも更新
        """
        added_index = int(self.get_selected_atom_indices()[0]) #Selected atoms内の1番最初のindexだけ取得
        # self.f_add_functionalgroup(added_index=added_index)
        
        self.atoms_base_addmole = smiles_to_ase_atoms(self.vh.addfunctional_smiles.value)
        self.index_remain_adv = 0 # atoms_base_addmoleの残す原子の初期index
        self.index_h_adv      = [i for i in f_neighbor_atom(self.atoms_base_addmole, self.index_remain_adv) if self.atoms_base_addmole.symbols[i]=="H"][0] # 残す原子に結合しているH原子のindex

        # ---addmolのviewで表示するAtoms更新 ---#
        self.addmol_view.remove_component(self.addmol_component)                                           # addmol_viewに追加したAtomsを削除
        self.addmol_component = self.addmol_view.add_component(nv.ASEStructure(self.atoms_base_addmole))   # addmol_viewに追加するAtoms
        
        # addmoleの消す水素原子のindexをオレンジ色に表示
        selection_str=f'@{self.index_h_adv}'
        self.addmol_view.add_representation("ball+stick",selection=selection_str,color="darkorange",aspectRatio=4)
        
        
    def addvalence(self, clicked_button: Button):
        self.append_tmpatoms_to_atomslist(self.atoms)
        atom_indices = self.get_selected_atom_indices()
        for _index in atom_indices:
            # 新しく作る結合の推定共有結合距離の取得
            bond_length = get_covalent_radii(self.vh.addvalence_symbol.value, self.atoms.symbols[_index])
            # 距離が離れているときに原子追加
            self.atoms.append(Atoms(self.vh.addvalence_symbol.value, positions=[get_far_position(self.atoms, _index, distance=bond_length)])[0])
        
            self.estimator  = Estimator(calc_mode=self.calc_mode, model_version=self.model_version)
            self.atoms.calc = ASECalculator(self.estimator)      
            
        if self.vh.miniopt_checkbox.value:   # checkbox==Trueだったら少しだけopt
            opt = FIRELBFGS(self.atoms, maxstep_fire=0.5, maxstep_lbfgs=0.5, logfile=None)
            opt.run(fmax=0.5,steps=5)
            
        self.update_display()
  
    def insert_molecule(self, clicked_button: Button):
        """分子を指定位置に挿入
        """
        self.append_tmpatoms_to_atomslist(self.atoms)
        atom_indices = self.get_selected_atom_indices()
        
        smiles_str       = self.vh.insertMol_smiles.value
        direction_str    = self.vh.insertMol_direction.value
        position_float   = float(self.vh.insertMol_position.value)
        
        smiles_str = smiles_str.strip(" ")  
        
        #smilesにHが入力されたときは水素挿入、H以外だった場合はそのsmilesを挿入
        if smiles_str=="H" :
            insert_smiles = Atoms("H", positions=[[0,0,0]])
        else :
            # insert_smiles = pubchem_atoms_search(smiles=smiles_str)
            insert_smiles = smiles_to_ase_atoms(smiles_str)
            
        insert_smiles_pos = insert_smiles.get_positions()
        
        #smilesで取得した座標を指定した数値だけx,y,z軸の方向に動かす
        if direction_str   == "x+" : 
            insert_smiles_pos = insert_smiles_pos+np.array([position_float,0,0]) #npは2行目以降の足りない行にも[0,0,1]を足すことができる
        elif direction_str == "x-" : 
            insert_smiles_pos = insert_smiles_pos+np.array([-position_float,0,0])
        elif direction_str == "y+" : 
            insert_smiles_pos = insert_smiles_pos+np.array([0,position_float,0])
        elif direction_str == "y-" : 
            insert_smiles_pos = insert_smiles_pos+np.array([0,-position_float,0])
        elif direction_str == "z+" : 
            insert_smiles_pos = insert_smiles_pos+np.array([0,0,position_float])
        elif direction_str == "z-" : 
            insert_smiles_pos = insert_smiles_pos+np.array([0,0,-position_float])
        else : 
            raise ValueError("Unexpected button", clicked_button.description)
        
        """ここにsmilesの重心を考慮したスクリプトを入れたい
        """
        
        #Selected atomsで指定した原子の上に分子をくっつける
        for atom_index in atom_indices :
            specified_atom_pos = self.atoms[atom_index].position.copy()
            insert_final_pos = insert_smiles_pos+specified_atom_pos
            
            insert_smiles.positions = insert_final_pos
            for i in insert_smiles :
                self.atoms.append(i)
    
        self.update_display()
        
        
    def redo(self, clicked_button: Button):
        try : 
            next_atoms=self.redo_list.pop()
            self.undo_list.append(self.atoms.copy())
            self.estimator = Estimator(calc_mode=self.calc_mode, model_version=self.model_version)
            self.undo_list[-1].calc = ASECalculator(self.estimator)
        except : 
            print("Can't Redo")
        else:
            # atomsの原子数を操作の一個前と合わせている (原子数が1個前のatomsの方が多ければ現在のatomsに水素原子をappend、少なければpopで原子数を強引に合わせている)
            if len(self.atoms)<len(next_atoms):
                hydrogen_atom = Atom(symbol="H", position=[0,0,0])
                for i in range(len(next_atoms)-len(self.atoms)): 
                    self.atoms.append(hydrogen_atom)
            elif len(self.atoms)>len(next_atoms):
                for i in range(len(self.atoms)-len(next_atoms)):
                    self.atoms.pop()
            else:
                pass
            
            self.atoms.symbols = next_atoms.get_chemical_symbols()
            self.atoms.positions = next_atoms.get_positions()
            
        self.update_display()    
        
    def undo(self, clicked_button: Button):
        try : 
            prev_atoms=self.undo_list.pop()
            self.redo_list.append(self.atoms.copy())
            self.estimator = Estimator(calc_mode=self.calc_mode, model_version=self.model_version)
            self.redo_list[-1].calc = ASECalculator(self.estimator)
        except : 
            print("Can't Undo")
        else:
            # atomsの原子数を操作の一個前と合わせている (原子数が1個前のatomsの方が多ければ現在のatomsに水素原子をappend、少なければpopで原子数を強引に合わせている)
            if len(self.atoms)<len(prev_atoms):
                hydrogen_atom = Atom(symbol="H", position=[0,0,0])
                for i in range(len(prev_atoms)-len(self.atoms)): 
                    self.atoms.append(hydrogen_atom)
            elif len(self.atoms)>len(prev_atoms):
                for i in range(len(self.atoms)-len(prev_atoms)):
                    self.atoms.pop()
            else:
                pass
            
            self.atoms.symbols = prev_atoms.get_chemical_symbols()
            self.atoms.positions = prev_atoms.get_positions()
            
        self.update_display()        
        
        
    def setcalc(self, clicked_button: Button):
        self.calc_mode = self.vh.setcalc_calculator.value
        self.model_version = self.vh.setcalc_version.value
        self.estimator = Estimator(calc_mode=self.calc_mode, model_version=self.model_version)
        self.atoms.calc=ASECalculator(self.estimator)
        print(f"Set calculator {self.calc_mode}, {self.model_version}")
        self.update_display()
        
    def clear_textarea(self, clicked_button: Button):
        self.vh.selected_atoms_textarea.value=""
        self.show_one_part_index() #show index in Selected atomsにチェックボックスが入っていた場合のみSelected atoms内の原子のlabelを追加

    def exchange_index(self, clicked_button: Button):
        """selected atoms内の2つのindex番号を変える
        """
        self.append_tmpatoms_to_atomslist(self.atoms)
        atom_indices = self.get_selected_atom_indices()
        try : 
            num_1 = atom_indices[0]
            num_2 = atom_indices[1]
        except :
            print("Can't exchange")
        else :
            old_mol = self.atoms.copy()
            old_mol_symbol= old_mol.symbols
            old_mol_pos   = old_mol.get_positions()

            self.atoms[num_1].symbol   = old_mol_symbol[num_2]
            self.atoms[num_1].position = old_mol_pos[num_2]
            self.atoms[num_2].symbol   = old_mol_symbol[num_1]
            self.atoms[num_2].position = old_mol_pos[num_1]
            print(f"Exchange indices #{num_1} and #{num_2}")
            self.update_display()
    
    def setconstraints(self, clicked_button: Button):
        del self.atoms.constraints
        atom_indices = self.get_selected_atom_indices()
        
        if len(atom_indices)==0 : 
            print(f"Delete constraints")
        else : 
            const_num = FixAtoms(indices=atom_indices)
            self.atoms.set_constraint(const_num)
            print(f"Set constraints for {const_num}")
        
    def delconstraints(self, clicked_button: Button):
        del self.atoms.constraints
        print(f"Delete constraints")

##########################################################################
        
    def move(self, clicked_button: Button):
        self.append_tmpatoms_to_atomslist(self.atoms)
        a = self.vh.move.value

        for index in self.get_selected_atom_indices():
            if clicked_button.description == "X+":
                self.atoms[index].position += [a, 0, 0]
            elif clicked_button.description == "X-":
                self.atoms[index].position -= [a, 0, 0]
            elif clicked_button.description == "Y+":
                self.atoms[index].position += [0, a, 0]
            elif clicked_button.description == "Y-":
                self.atoms[index].position -= [0, a, 0]
            elif clicked_button.description == "Z+":
                self.atoms[index].position += [0, 0, a]
            elif clicked_button.description == "Z-":
                self.atoms[index].position -= [0, 0, a]
        self.update_display()
    
    def move_specified(self, clicked_button: Button):
        """指定したベクトル方向に平行移動
        """ 
        self.append_tmpatoms_to_atomslist(self.atoms)
        num0=self.vh.specified_vec0.value
        num1=self.vh.specified_vec1.value
        mol_pos=self.atoms.get_positions()
        mol_pos_num0 = mol_pos[num0]
        mol_pos_num1 = mol_pos[num1]
        axis_vec=mol_pos_num1-mol_pos_num0 #ベクトルを取得
        axis_vec_unit = axis_vec/np.linalg.norm(axis_vec) #単位ベクトルに変換
        axis_vec_move = axis_vec_unit*self.vh.move.value  #移動させる大きさのベクトルに変換
        
        if clicked_button.description == "Move -": #-が押されていたら動かす座標を-1倍
            axis_vec_move*=-1
        
        for index in self.get_selected_atom_indices():
            self.atoms[index].position += axis_vec_move
        self.update_display()

    def rotate(self, clicked_button: Button):
        self.append_tmpatoms_to_atomslist(self.atoms)
        atom_indices = self.get_selected_atom_indices()
        deg = self.vh.rotate.value
        temp = self.atoms[atom_indices]

        if clicked_button.description == "X+":
            temp.rotate(deg, "x", center="COP")
        elif clicked_button.description == "X-":
            temp.rotate(-deg, "x", center="COP")
        elif clicked_button.description == "Y+":
            temp.rotate(deg, "y", center="COP")
        elif clicked_button.description == "Y-":
            temp.rotate(-deg, "y", center="COP")
        elif clicked_button.description == "Z+":
            temp.rotate(deg, "z", center="COP")
        elif clicked_button.description == "Z-":
            temp.rotate(-deg, "z", center="COP")
        rotep = temp.positions
        for i, atom in enumerate(atom_indices):
            self.atoms[atom].position = rotep[i]
        self.update_display()
    ##############################################################
    def rotate_specified_vec(self, clicked_button: Button):
        """指定したベクトルを回転軸として指定
        """
        self.append_tmpatoms_to_atomslist(self.atoms)
        atom_indices = self.get_selected_atom_indices()
        deg = self.vh.rotate.value
        temp = self.atoms[atom_indices]

        num0=self.vh.specified_vec0.value
        num1=self.vh.specified_vec1.value
        mol_pos=self.atoms.get_positions()
        mol_pos_num0 = mol_pos[num0]
        mol_pos_num1 = mol_pos[num1]
        axis_vec=mol_pos_num1-mol_pos_num0
        
        if clicked_button.description == "Rotate +": pass #Rotate +と-でdegreeを変更
        else : deg=-deg
            
        try : temp.rotate(-deg, axis_vec,center=self.atoms[num0].position)
        except : 
            print(f"[Error] Can't rotate")
        else :
            rotep = temp.positions
            for i, atom in enumerate(atom_indices):
                self.atoms[atom].position = rotep[i]

            rotep = temp.positions
            for i, atom in enumerate(atom_indices):
                self.atoms[atom].position = rotep[i]
            self.update_display()

    ###############################################################
        
#     def run_opt(self, clicked_button: Button):
#         """OPT only specified steps and FIX atoms if NOT in text atoms list"""
#         self.append_tmpatoms_to_atomslist(self.atoms)
#         if self.vh.constraint_checkbox.value:
#             # Fix non selected atoms. Only opt selected atoms.
#             print("Opt with selected atoms: fix non selected atoms")
#             atom_indices = self.get_selected_atom_indices()
#             constraint_atom_indices = [
#                 i for i in range(len(self.atoms)) if i not in atom_indices
#             ]
#             self.atoms.set_constraint(FixAtoms(indices=constraint_atom_indices))
#         opt = LBFGS(self.atoms, maxstep=0.04, logfile=None)
#         steps: Optional[int] = self.vh.opt_step.value
#         if steps < 0:
#             steps = None  # When steps=-1, opt until converged.
#         opt.run(fmax=self.vh.opt_fmax.value, steps=steps)
        
#         """constraintsを消すのを追加
#         """
#         del self.atoms.constraints
#         print("delete constraints")
        
#         print(f"Run opt for {steps} steps")
#         self.update_display()
        
    def run_opt(self, clicked_button: Button):
        """OPT only specified steps and FIX atoms if NOT in text atoms list"""
        self.append_tmpatoms_to_atomslist(self.atoms)

        # 指定した原子だけoptしたい場合、一時的にconstをかける(もともとのconstも考慮)
        if self.vh.constraint_checkbox.value:   
            # Fix non selected atoms. Only opt selected atoms.
            print("Opt with selected atoms: fix non selected atoms")
            atom_indices = self.get_selected_atom_indices()
            constraint_atom_indices = [i for i in range(len(self.atoms)) if i not in atom_indices]
            constraint_atom_indices = list(set(constraint_atom_indices)|set(self.original_cons))
            self.atoms.set_constraint(FixAtoms(indices=constraint_atom_indices))
            # print(constraint_atom_indices)
        
        # step数取得
        steps: Optional[int] = self.vh.opt_step.value
        if steps < 0:
            steps = None  # When steps=-1, opt until converged.

        # optのログをTextareaに #################
        self.vh.optlog_textarea.value = f'{"-"*57}\n'
        self.vh.optlog_textarea.value += f'model_version :  {self.model_version}\n'
        self.vh.optlog_textarea.value += f'calc_mode     :  {self.calc_mode}\n'
        self.vh.optlog_textarea.value += f'{"-"*57}\n'

        # opt100stepごとに表示するoutputの設定
        def optshow():
            now_fmax = (opt.atoms.get_forces() ** 2).sum(axis=1).max()**0.5
            now_fmax = f"{now_fmax:.4f}"
            now_ene = opt.atoms.get_potential_energy()
            now_ene = f"{now_ene:.6f}"
            if self.vh.opt_algorithm_dropdown.value=="FIRELBFGS":
                self.vh.optlog_textarea.value += f"\nFIRELBFGS: {str(opt.nsteps).rjust(7)}   {str(now_ene).rjust(15)} eV    {str(now_fmax).rjust(11)} eV/A"
            elif self.vh.opt_algorithm_dropdown.value=="LBFGS":
                self.vh.optlog_textarea.value += f"\nLBFGS:     {str(opt.nsteps).rjust(7)}   {str(now_ene).rjust(15)} eV    {str(now_fmax).rjust(11)} eV/A"
            else:
                self.vh.optlog_textarea.value += f"\nFIRE:      {str(opt.nsteps).rjust(7)}   {str(now_ene).rjust(15)} eV    {str(now_fmax).rjust(11)} eV/A"
        self.vh.optlog_textarea.value += f"{' '*10} {'Step'.rjust(7)} {'Energy'.rjust(17)} {'fmax'.rjust(17)}"

        # opt
        per_output=20
        if self.vh.opt_algorithm_dropdown.value=="FIRELBFGS":
            opt = FIRELBFGS(self.atoms,logfile=None, maxstep_lbfgs=self.vh.opt_maxstep.value)
        elif self.vh.opt_algorithm_dropdown=="LBFGS":
            opt = LBFGS(self.atoms,logfile=None, maxstep=self.vh.opt_maxstep.value)
        else:
            opt = FIRE(self.atoms,logfile=None, maxstep=self.vh.opt_maxstep.value)
        opt.attach(optshow,per_output)
        opt.run(steps=steps, fmax=self.vh.opt_fmax.value)

        # optが収束したときと、stepsに達したときにもエネルギーとfmaxを出力
        if (opt.nsteps > 1 and opt.converged()) or (opt.nsteps==steps and opt.nsteps%per_output != 0) : optshow()
        #######################################
        
        # もともとのconstに戻す
        if len(self.original_cons)==0:   # constが空の場合にelseと同じ掛け方をすると[FixAtoms(indices=[])]となるから
            del self.atoms.constraints
        else:
            self.atoms.set_constraint(FixAtoms(self.original_cons)) 
            
        # if self.vh.optlog_checkbox.value:
        #     print(f"Run opt for {steps} steps and fmax = {self.vh.opt_fmax.value}")
        self.update_display()

    def download_image(self, clicked_button: Optional[Button] = None):
        filename = self.vh.filename_text.value
        self.v.download_image(filename=filename)
    ################################################################################
    def download_transparent_image(self, clicked_button: Optional[Button] = None):
        filename = self.vh.filename_text.value
        self.v.download_image(filename=filename, transparent=True)
    ################################################################################
    def save_image(self, clicked_button: Optional[Button] = None):
        filename = self.vh.filename_text.value
        save_image(filename, self.v)
        
    def show_force_event(self, event: Bunch):
        self.show_force = self.vh.show_force_checkbox.value
        self.update_display()
        
    def show_axes_event(self, event: Bunch):
        self.show_axes = self.vh.show_axes_checkbox.value
        self.update_display()

    def show_index_event(self, event: Bunch):
        """self.vh.show_indexは全原子を、
           self.vh.show_index_one_partはSelected Atoms内の原子だけのindexを追加
        """
        
        '''チェックボックスで場合分け
        '''
        if self.vh.show_index_one_part_checkbox.value==True : #この行のifはshow_one_part_index関数の中にも同じif文が入っているので不要だが分かりやすくするために記載(下のelifは必要)
            self.show_one_part_index()
        elif (self.vh.show_index_one_part_checkbox.value==False) & (self.vh.show_index_checkbox.value==True):
            self.v.remove_label()
            self.show_all_index()
        else :
            self.v.remove_label()

        self.update_display()
        
    def show_one_part_index(self): #ボタン関係ないただの関数。show_index_eventの中や範囲選択のスライダーで使う
        '''show index in Selected atomsにチェックボックスが入っていた場合のみSelected atoms内の原子のlabelを追加
        '''
        atom_indices = self.get_selected_atom_indices()
        label_indices=[] 
        for i in range(len(self.atoms)):
            if i in atom_indices : label_indices.append(i)
            else : label_indices.append("")
        label_indices=[str(i) for i in label_indices]
        if self.vh.show_index_one_part_checkbox.value==True : 
            self.v.remove_label()
            self.v.add_label(
                color="blue", labelType="text",
                labelText=label_indices,
                zOffset=1.0, attachment='middle_center', radius=1.0
            )
            
    def show_all_index(self): #ボタン関係ないただの関数。全原子のindexを可視化     
        '''ここから4行はSelected Atoms内の原子を取得
        '''
        atom_indices = self.get_selected_atom_indices()
        self.v.add_label(
            color="black", labelType="text",
            labelText=[str(i) for i in range(self.atoms.get_global_number_of_atoms())],
            zOffset=1.0, attachment='middle_center', radius=1.0
        )
        
        
    def on_accordion_change(self,accordion_change):
        """ accordionが開くとclicked modeが対応したモードに変更
        """
        if accordion_change.new==1:
            self.vh.touch_mode_select.value="Add molecule"
            self.addmol_view.handle_resize()   # addmol_viewを再表示ししている(クラスにviewerを組み込むと最初表示されないバグの対処)
            
        elif accordion_change.new==2:
            self.vh.touch_mode_select.value="Add Valence"
        elif accordion_change.new==0:
            self.vh.touch_mode_select.value="Delete"
        else:
            self.vh.touch_mode_select.value="Normal"
        
    def on_tab_change(self,tab_change):
        """ 3番目のタブ(=Additionタブ)以外が選択されたときにタブを閉じるようにする
        """
        if tab_change.new!=3:
            self.vh.accordion_add.selected_index=None
            
    def on_dropdown_add_image_buttons_change(self,image_buttons_change):
        """ addmoleの配位数のdropdownが変更された時に実行される関数
        """
        if image_buttons_change['type'] == 'change' and image_buttons_change['name'] == 'value':
            self.f_addmole_str_to_atoms()
            self.index_remain_adv = 0 # atoms_base_addmoleの残す原子の初期index
            self.index_h_adv      = [i for i in f_neighbor_atom(self.atoms_base_addmole, self.index_remain_adv) if self.atoms_base_addmole.symbols[i]=="H"][0] # 残す原子に結合しているH原子のindex

            # ---addmolのviewで表示するAtoms更新 ---#
            self.addmol_view.remove_component(self.addmol_component)                                           # addmol_viewに追加したAtomsを削除
            self.addmol_component = self.addmol_view.add_component(nv.ASEStructure(self.atoms_base_addmole))   # addmol_viewに追加するAtoms
            
            # addmoleの消す水素原子のindexをオレンジ色に表示
            selection_str=f'@{self.index_h_adv}'
            self.addmol_view.add_representation("ball+stick",selection=selection_str,color="darkorange",aspectRatio=4)
            
    def f_addmole_str_to_atoms(self):
        """ addmolのボタンで選択した原子と配位数のatomsを作成する関数
        """
        # bond_typeごとで読み込むjsonファイルを変更
        if self.vh.add_image_buttons.value=="1-coordination":
            self.atoms_base_addmole = read(f"{self.folder_name_addmole}/atoms_mono_coord.json")
        elif self.vh.add_image_buttons.value=="2-coordination":
            self.atoms_base_addmole = read(f"{self.folder_name_addmole}/atoms_di_coord.json")
        elif self.vh.add_image_buttons.value=="3-coordination":
            self.atoms_base_addmole = read(f"{self.folder_name_addmole}/atoms_tri_coord.json")
        else :
            self.atoms_base_addmole = read(f"{self.folder_name_addmole}/atoms_tetra_coord.json")
            
        # 0番のindexを指定の元素に置換してopt
        self.atoms_base_addmole.symbols[0]=self.addmole_mol
        self.estimator = Estimator(calc_mode=self.calc_mode, model_version=self.model_version)
        self.atoms_base_addmole.calc=ASECalculator(self.estimator)
        opt = FIRELBFGS(self.atoms_base_addmole,logfile=None)
        opt.run(fmax=0.5,steps=5)
            
            
        
    def f_addmole(self,added_index):
        """ addmole関数
        """
        
        # addmoleで使用する引数の取得
        self.append_tmpatoms_to_atomslist(self.atoms)   # addmole前のatomsをredoundoできるように保存
        # added_index = int(self.get_selected_atom_indices()[0]) #Selected atoms内の1番最初のindexだけ取得
        if self.atoms.symbols[added_index]!="H":
            print("Warning : picked index is not H atoms")

        # 新しく作る結合の推定共有結合距離の取得
        bond_length = get_covalent_radii(self.addmole_mol, self.atoms.symbols[f_neighbor_atom(self.atoms,added_index)[0]])
        
        # add_mole関数使用            
        indx_sbs=[f_neighbor_atom(self.atoms, added_index)[0], added_index]
        indx_ads=[self.index_h_adv, self.index_remain_adv]
        
        self.atoms=add_mole_editor(self.atoms, self.atoms_base_addmole, indx_sbs, indx_ads, a_bond_length=bond_length)
        self.estimator = Estimator(calc_mode=self.calc_mode, model_version=self.model_version)
        self.atoms.calc=ASECalculator(self.estimator)
        
        # checkbox==Trueだったら少しだけopt
        if self.vh.miniopt_checkbox.value:   
            opt = FIRELBFGS(self.atoms, maxstep_fire=0.5, maxstep_lbfgs=0.5, logfile=None)
            opt.run(fmax=0.5,steps=10)
    
        # 一部のindexを表示する場合は"selected atoms"に入っている番号も更新
        if self.vh.show_index_one_part_checkbox.value==True:
            self.set_atoms()
        # 全indexを表示させている場合は"selected atoms"に入っている番号は更新せずにviewerのindexだけ更新。(selected atomsを変えている場合は更新したくない場合があるため)
        elif self.vh.show_index_checkbox.value==True:
            self.show_index_event(event=None)
        else :
            pass
        
        self.update_display()   
        