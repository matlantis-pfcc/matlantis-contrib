import numpy as np
import sys,IPython,ase,spglib
from ipywidgets import  widgets
from ipywidgets import (Button, Checkbox, FloatSlider, GridspecLayout, HBox,
                        IntSlider, Label, Output, Text, Textarea)
from ase import Atoms
import nglview as  nv
import pandas as pd
from tqdm.auto import tqdm

from pymatgen.analysis.phase_diagram import PDEntry
from pymatgen.core.composition import Composition
from pymatgen.analysis.phase_diagram import PhaseDiagram, PDPlotter

import plotly.graph_objects as go
import matplotlib.figure as figure
import matplotlib.pyplot as plt

from pfp_api_client.pfp.estimator import Estimator, EstimatorCalcMode
from pfp_api_client.pfp.calculators.ase_calculator import ASECalculator

estimator = Estimator(model_version="v3.0.0" ,calc_mode=EstimatorCalcMode.CRYSTAL)
df = pd.read_pickle("input/mp.gz")

class Crystalsearch:
    """search crystal from df"""
    def __init__(self, df=df, estimator  = estimator): 
        self.estimator = estimator
        self.df =df

        #周期表
        atomsdict = ase.data.atomic_numbers.copy()
        atomsdict.pop("X")
        self.atomsdict = atomsdict
        periodicgrid = widgets.GridspecLayout(10,18,width='860px')  
        col,row = 0,0
        griddict = {}
        for tag, i in atomsdict.items():
            if i == 2: col = 17;  # He
            if i == 3: row = 1; col = 0# Li
            if i == 5: col = 12 # B
            if i == 11: row = 2; col = 0 #/ Na
            if i == 13: col = 12 #// Al
            if i ==  57: row = 8; col = 2# // La
            if i == 72: row = 5; col = 3 # // Hf
            if i == 89: row = 9; col = 2  #// Ac
            if i == 104: row = 6; col = 3 #// Rf
            x  =   tag in estimator.supported_elements()
            periodicgrid[row, col] = widgets.ToggleButton(name = tag, description=tag,layout=widgets.Layout(width='42px', height='22px'))
            periodicgrid[row, col].observe(self.update_selection, 'value')
            griddict[tag] = [row,col]
            if (col == 17): row+=1;col = 0;
            else : col+=1
                
        self.periodicgrid  = periodicgrid #グリッドウェジット
        self.griddict  =  griddict #原子番号から場所を特定するリスト
        self.selected_atoms = []        
        
        #コントロールボックスの準備
        buttonlayout = widgets.Layout(width='80px', height='27px')
        inputlayout = widgets.Layout(width='175px', height='27px')

        self.versionbox = widgets.Dropdown(description="version",options=estimator.available_models[::-1],layout=inputlayout)
        self.supportbutton = widgets.Button(description="support",layout=buttonlayout)
        self.supportbutton.on_click(self.show_support)

        self.searchbutton = widgets.Button(description="search" ,layout=buttonlayout)
        self.searchbutton.on_click(self.search)
        
        self.phasebutton = widgets.Button(description="hullview" ,layout=buttonlayout)
        self.phasebutton.on_click(self.phase_diagram)
        
        self.clearbutton = widgets.Button(description="clear" ,layout=buttonlayout)
        self.clearbutton.on_click(self.clear)
        
        self.anyonlybox = widgets.Dropdown(description="ele option",options=["within","any","only"], layout=inputlayout)
        self.hullbox    = widgets.BoundedFloatText(value=0.3, min=0, step=0.05, description='hullabove<', layout=inputlayout ) 
        self.contbox    = widgets.HBox( [self.anyonlybox,self.hullbox,self.searchbutton,self.phasebutton,self.clearbutton, self.versionbox,self.supportbutton])
        
        #ボタンのスタイルを設定（JAVASCRIPT機能）
        display(IPython.display.HTML('<style> .widget-gridbox {background-color: #222266;padding: 3px} .widget-toggle-button {padding: 1px; font-size: 12px;border-radius: 6px;}</style>'))
        self.mpsearch = widgets.VBox([self.periodicgrid,self.contbox])
        display(self.mpsearch)

        #検索結果、NGLVIEW、HULL等はじめ非表示部分の準備
        self.nglbox = widgets.Box(layout= widgets.Layout(width='400px', height='350px') )        
        self.nglbutton = widgets.Button(description="view" ,layout=buttonlayout)
        self.nglbutton.on_click(self.showngl)
        self.rowslider=widgets.IntSlider(value = 1, min=0,max= 3000, step=1,description='No')
        self.rowslider.observe(self.showdf)        
        self.dfoutput  = widgets.Output(layout={'border': '1px solid black'})
        self.dfbox    = widgets.HBox( [self.nglbox, widgets.VBox( [ widgets.HBox([self.nglbutton,self.rowslider,]) ,self.dfoutput] )] )
        
        self.phasebox = widgets.Box(layout= widgets.Layout( height='450px') )
        self.showcol = ["material_id","pretty_formula","energy","e_above_hull"]
        self.primitiveatomslist = []
        self.conventionalatomslist =[]
    
    #周期表内のトグルをクリックしたときの動作　（表示およびリストの更新）
    def update_selection(self , change):
        owner = change['owner']
        name = owner.description
        if change['new']:
            owner.icon = 'check'
            self.selected_atoms += [name]
        else:
            owner.icon = ""
            self.selected_atoms.remove(name)
    
    #PFPバージョンに応じたサポート原子のチェック更新　
    def show_support(self,change):
        support = self.estimator.supported_elements(model_version = self.versionbox.value)
        for ele in self.atomsdict:
            self.periodicgrid[self.griddict[ele]].value = ele in support
    
    def clear(self,change):
        self.dfoutput.clear_output()
        self.nglbox.children =[]
        self.phasebox.children =[]
        self.ngl = None    
        for ele in self.atomsdict:
            self.periodicgrid[self.griddict[ele]].value = False
        self.mpsearch.children = [self.periodicgrid,self.contbox]
    
    def search(self,change):
        self.dfoutput.clear_output()
        self.nglbox.children = []
        cands  = select_element(df = self.df, elements = self.selected_atoms, eleselect = self.anyonlybox.value )
        cands  = select_hull(cands, above_under = self.hullbox.value )
        self.cands = cands        
        self.rowslider.max=max(len(cands)-1,1)
        self.rowslider.value = 0
        self.mpsearch.children = [self.periodicgrid,self.contbox,self.dfbox] 
        self.showdf(self)

    def showdf(self,change):
        self.dfoutput.clear_output()
        with self.dfoutput:
            display(self.cands[self.showcol].iloc[self.rowslider.value : self.rowslider.value+10])

    def showngl(self,change):
        primitiveatoms = self.cands.iloc[self.rowslider.value].atoms
        conventionalatoms = prim_to_conv(primitiveatoms)
        r = [1,1,1] if len(conventionalatoms) > 30 else [2,2,2] if len(conventionalatoms) > 6 else  [3,3,3]
        v = nv.show_ase(conventionalatoms.repeat(r))
        v._remote_call("setSize",args=["350px", "350px"])
        v.add_unitcell()
        self.nglbox.children = [v]
        self.ngl = v
        self.primitiveatoms = primitiveatoms
        self.conventionalatoms = conventionalatoms
        self.primitiveatomslist += [primitiveatoms]
        self.conventionalatomslist += [conventionalatoms]
      
    def phase_diagram(self,change):
        if 2 <= len(self.selected_atoms) <=4:
            self.anyonlybox.value = "within"
            self.search(change)
            self.pd = get_phasediagram(self.cands)
            self.pdfig = get_phasediagram_fig(self.pd , show_unstable=self.hullbox.value + 0.001 )
            self.phasebox.children = [self.pdfig]
            self.mpsearch.children = [self.periodicgrid,self.contbox,self.dfbox,self.phasebox]
            
    def get_pfp_df(self):
        cands = self.cands
        mv = self.versionbox.value
        print(f"calc {len(cands)} cands by pfp {mv}")
        estimator = Estimator(model_version = mv ,calc_mode=EstimatorCalcMode.CRYSTAL)
        estimator_u0 = Estimator(model_version = mv ,calc_mode=EstimatorCalcMode.CRYSTAL_U0)
        calculator = ASECalculator(estimator)
        calculator_u0 = ASECalculator(estimator_u0)
        n_hubbard = cands["is_hubbard"].sum()
        if n_hubbard > 0:
            print(f"[WARNING] {n_hubbard} / {len(cands)} atoms contain Hubbard U correction, its U parameter might be different between Materials project & PFP. Please check carefully.")
        pfpene ,pfpfmax ,lenatoms = [],[],[]
        for index,atoms,is_hubbard in tqdm(cands[["atoms","is_hubbard"]].itertuples(), total=len(cands)):
            if is_hubbard: 
                atoms.calc = calculator
                shift_energies=estimator.get_shift_energy_table()
            else:
                atoms.calc = calculator_u0
                shift_energies=estimator_u0.get_shift_energy_table()

            total_shift_energy = sum([shift_energies[i] for i in atoms.get_atomic_numbers()])
            pfpene += [ atoms.get_potential_energy() + total_shift_energy ]
            pfpfmax +=  [ round( ((((atoms.get_forces())**2).sum(axis=1).max())**0.5 ),5) ]
            lenatoms += [len(atoms)]
        
        cands["pfpene"] = pfpene
        cands["pfpfmax"] = pfpfmax
        cands["lenatoms"] = lenatoms
        self.cands = cands
        print("end")
        self.show_yy()
        return cands

    def show_yy(self):
        cands =self.cands
        title = str(self.selected_atoms)
        xlabel ="mp"
        ylabel ="pfp"
        unit ="ev/atom"
        x = cands.energy/ cands.lenatoms
        y = cands.pfpene/ cands.lenatoms

        MAE = np.average(np.absolute( x-y ))
        plt.figure(figsize=figure.figaspect(1))
        plt.title (f"{title} \n {xlabel} vs {ylabel}  MAE:{MAE:.2f} ")
        plt.scatter(x, y  ,label = f"{estimator.model_version}" , alpha = 1.0,s=10)
        plt.legend()
        YMax = max(list(y)+list(x))
        YMin = min(list(y)+list(x))
        YMax = YMax + (YMax - YMin)*0.1
        YMin = YMin - (YMax - YMin)*0.1
        plt.plot([YMin,YMax],[YMin,YMax], 'k-')
        plt.ylim(YMin,YMax)
        plt.xlim(YMin,YMax)
        plt.xlabel( xlabel + " " + unit)
        plt.ylabel( ylabel + " " + unit)
        plt.show()

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

def get_phasediagram(df,energy = "energy"):
    entries = list()
    for candsindex , structure , name, energy in df[["atoms","pretty_formula",energy]].itertuples():        
        composition = Composition(structure.get_chemical_formula())
        pde = PDEntry(composition, energy, name=name)
        pde.entry_id = candsindex
        entries.append(pde)
    return PhaseDiagram(entries)

def get_phasediagram_fig(pd , show_unstable=1 ):
    plotter = PDPlotter(pd, show_unstable = show_unstable)
    fig =plotter.get_plot()
    fig.update_layout(width= 800 ,height = 400 ,paper_bgcolor="rgba(0.9.,0.9.,0.9.,0.9)")
    for x in fig.data:
        if hasattr(x, 'marker'):x.marker.size = 4
    fig = go.FigureWidget(data=fig)    
    return fig