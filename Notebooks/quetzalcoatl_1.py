import functools
import subprocess
import requests
import json
from ast import literal_eval
from pylab import *
from scipy.signal import argrelextrema
import pandas as pd
import glob
from rdkit import Chem
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import Draw
IPythonConsole.ipython_useSVG=True 
import ipywidgets as widgets
import pandas as pd
import numpy as np
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
from rdkit import Chem
from rdkit.Chem import AllChem
import py3Dmol
from ipywidgets import widgets, interact,fixed
from IPython.display import display
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# Generating the animation file
def Animator_data(ID):
    # Opening JSON file
    f = open('crd.json')
 
    # returns JSON object as
    # a dictionary
    data = json.load(f)

    element_symbols = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
    'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th',
    'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og']

    
    
    freq = data[ID]["key_value_pairs"]["frequencies_cm"]
    freq= freq.replace("[", " ")
    freq=freq.replace("]", " ")
    freq=freq.replace("\n", " ")
    freq=freq.split()
    freqs_db = [round(float(i),2) for i  in freq ] 
    
    eq_pos = reshape(data[ID]["positions"]["__ndarray__"][2],\
                     (data[ID]["positions"]["__ndarray__"][0][0],\
                      data[ID]["positions"]["__ndarray__"][0][1]))
    at_num = data[ID]["numbers"]['__ndarray__'][2]
    elements = []
    for i in at_num:
        elements.append(element_symbols[i-1])
    
    modes = data[ID]["data"]["eigenvec"]
    
    with open('readme_solids.txt', 'w') as f:

        f.write("[Molden Format]")
        f.write('\n')
        f.write('\n')

        f.write("[FREQ]")
        f.write('\n')

        for line in freqs_db:
            f.write(str(line))
            f.write('\n')

        f.write('\n')
        f.write("[FR-COORD]")
        f.write('\n')
        for i in range(0,len(elements)):
            f.write(str(elements[i]))
            f.write('       ')
            f.write(str('%.8f' % round(eq_pos[i][0],8)))
            f.write('       ')
            f.write(str('%.8f' % round(eq_pos[i][1],8)))
            f.write('       ')
            f.write(str('%.8f' % round(eq_pos[i][2],8)))
            f.write('\n')
        f.write('\n')
        f.write("[FR-NORM-COORD]")
        f.write('\n')
        for vib in range(0,len(modes)):
            f.write("vibration {0}".format(vib+1))
            f.write('\n')
            for el in range(0,len(elements)):
                f.write('       ')
                f.write(str('%.8f' % round(modes[vib][el][0],8)))
                f.write('       ')
                f.write(str('%.8f' % round(modes[vib][el][1],8)))
                f.write('       ')
                f.write(str('%.8f' % round(modes[vib][el][2],8)))
                f.write('\n')

# Auxiliar functions

def g(wavenumb_sweep, intensity_max, wavenumber_max, σ):
    G = intensity_max / (σ *sqrt(2 * pi)) * exp(-(wavenumb_sweep-wavenumber_max)**2 / (2*σ**2))
    new_y=array(G)  
    return new_y

def Gaus_norm( list_freq,list_int, sigma):
    ### Gausian function to broaden peaks
    wavenumb       = list(linspace(0,4000,10001))
    int_dft_m      = list_int + list(zeros(len(wavenumb)))
    x_dft          = list_freq + wavenumb
    
    all= []
    for i in range(len(int_dft_m )):
        all.append(tuple((x_dft[i],int_dft_m[i])))
    
    all.sort()
    x= []
    y= []
    for i in all:
        x.append(round(i[0],2))
        y.append(i[1])
    
    frequencies = x
    intensities = y

    
    # Crear un diccionario para realizar un seguimiento de las sumas de intensidades por frecuencia
    frequency_intensity_dict = {}
    
    for frequency, intensity in zip(frequencies, intensities):
    
        if frequency in frequency_intensity_dict:
            frequency_intensity_dict[frequency] += intensity
        else:
           
            frequency_intensity_dict[frequency] = intensity
    
    result_frequencies_dft = list(frequency_intensity_dict.keys())
    result_intensities_dft = list(frequency_intensity_dict.values())    
 
    
    wavenumb_dft   = result_frequencies_dft
    max_int    = max(result_intensities_dft)
    
    norm_int = []
    for i in result_intensities_dft:
        norm_int.append(i/max_int)
    
    ### Gausian function to broaden peaks
    pos_max    = argrelextrema(array(result_intensities_dft), np.greater)
    
    x          = array(wavenumb_dft)
    
    all_curve = 0
    σ = sigma
    for i in pos_max[0]:
        all_curve += g(x, norm_int[i], result_frequencies_dft[i],σ)      
    broad_int = all_curve   
    
    ### Normalization
    
    max_y = max(broad_int)
    int_norm_dft = []
    
    for i in broad_int:
        int_norm_dft.append((i/max_y))
    return wavenumb_dft, int_norm_dft, result_intensities_dft

def intensity_raman(raman_tensor):
    """ Average a Raman-activity tensor to obtain a scalar
    intensity. """

    # This formula came from D. Porezag and M. R. Pederson, Phys. Rev.
    # B: Condens. Matter Mater. Phys., 1996, 54, 7830.
    
        
    if raman_tensor==None or raman_tensor==0:
        return 0
    else: 

        alpha = (
            (raman_tensor[0][0] + raman_tensor[1][1] + raman_tensor[2][2])
            / 3.0)
    
        beta_squared = 0.5 * (
            (raman_tensor[0][0] - raman_tensor[1][1]) ** 2
            + (raman_tensor[0][0] - raman_tensor[2][2]) ** 2
            + (raman_tensor[1][1] - raman_tensor[2][2]) ** 2
            + 6.0 * (raman_tensor[0][1] ** 2 + raman_tensor[0][2] ** 2 +
                raman_tensor[1][2] ** 2)
            )
    
        return (45.0 * alpha ** 2 + 7.0 * beta_squared)
    
def spectra_data(ID, sigma_IR, sigma_Raman):
    
    f = open('crd.json')

    # returns JSON object as
    # a dictionary
    data = json.load(f)

    chemical_formula   = data[ID]["data"]["chemical_formula"]
    lattice_parameters = data[ID]["data"]["lattice_parameters"]
    lattice_angles     = data[ID]["data"]["lattice_angles"]
    cell_vectors       = array(data[ID]["cell"]["array"]["__ndarray__"][2]).reshape(3,3)
    mpid               = data[ID]["key_value_pairs"]["mpid"]
    density            = round(data[ID]["data"]["mass"]/data[ID]["data"]["volume"]*10**30*1.6726e-27,2)

    print("The chemical formula of this material is: ", chemical_formula)
    print("The lattice parameters of this material are, in the order (a,b,c) [A]: ", lattice_parameters)
    print("The lattice angles of this material are, in the order (α,β,γ): ", lattice_angles)
    print("The cell vectors of this material are, in matrix notation: ")
    print(cell_vectors)
    print(r"The density of this material in [kg/m^3]: ", density)
    print("For more properties of the material you can consult it in the materials project webpage with the following ID: ", mpid)

     #extracting raman spectra
    freq = data[ID]["key_value_pairs"]["frequencies_cm"]
    freq= freq.replace("[", " ")
    freq=freq.replace("]", " ")
    freq=freq.replace("\n", " ")
    freq=freq.split()
    freqs_db = [float(i) for i  in freq ]

    ints_ram_t = []
    for i in data[ID]["data"]["raman_tensors"]:
        ints_ram_t.append(intensity_raman(i))

    int_db = data[ID]["data"]["Ramanactive"]["__ndarray__"][2]
    freq_gn_raman, int_gn_raman, int_n_raman = Gaus_norm(freqs_db,ints_ram_t,sigma_Raman)
    # extracting ir spectra
    
    r          = requests.get('https://ramandb.oulu.fi/figs/irtable/{0}.json'.format(mpid))
    data_bytes = r.content
    data_s       = str(data_bytes)
    data_s       = data_s.replace("'", " ")
    data_s       = data_s.replace("\\n", " ")
    data_s       = data_s.replace("b", " ")
    data_s       = data_s.split()
    data_c       = [float(i) for i  in data_s ] 
    intensity    = []
    wave_l       = []


    for i in arange(0,len(data_c),2):
        intensity.append(data_c[i+1])
        wave_l.append(data_c[i])

    freq_gn_ir, int_gn_ir, int_n_ir = Gaus_norm(wave_l,intensity,sigma_IR)
    
    return freq_gn_ir, int_gn_ir, int_n_ir , freq_gn_raman, int_gn_raman, int_n_raman

# Generating 

def solids_spectra(filename ,freq_gn_ir, int_gn_ir,freq_gn_raman, int_gn_raman):
    
    def section(fle, begin, end):
        """
        yields a section of a textfile. 
        Used to identify [COORDS] section etc
        """
        with open(fle) as f:
            for line in f:
                # found start of section so start iterating from next line
                if line.startswith(begin):
                    for line in f: 
                        # found end so end function
                        if line.startswith(end):
                            return
                        # yield every line in the section
                        yield line.rstrip()    

    def parse_molden(filename='default.molden_normal_modes'):
        """
        Extract all frequencies, the base xyz coordinates 
        and the displacements for each mode from the molden file
        """
        all_frequencies = list(section(filename, '[FREQ]', '\n'))
        all_frequencies = [(float(freq),i) for i, freq in enumerate(all_frequencies)]
        coords = list(section(filename, '[FR-COORD]', '\n'))
        normal_modes = []
        for freq in range(len(all_frequencies)):
            if freq+1 != len(all_frequencies):
                normal_modes.append(list(section(filename, f'vibration {freq+1}', 'vibration')))
            else:
                normal_modes.append(list(section(filename, f'vibration {freq+1}', '\n')))
        return all_frequencies, coords, normal_modes

    def draw_normal_mode(mode=0, coords=None, normal_modes=None):
        """
        draws a specified normal mode using the animate mode from py3Dmol. 
        Coming from psi4 units need to be converted from a.u to A. 
        """
        fac=1  # bohr to A
        xyz =f"{len(coords)}\n\n"
        for i in range(len(coords)):
            atom_coords = [float(m) for m in  coords[i][8:].split('       ')]
            mode_coords = [float(m) for m in  normal_modes[mode][i][8:].split('       ')]
            xyz+=f"{coords[i][0:4]} {atom_coords[0]*fac} {atom_coords[1]*fac} {atom_coords[2]*fac} {mode_coords[0]*fac} {mode_coords[1]*fac} {mode_coords[2]*fac} \n"
        view = py3Dmol.view(width=400, height=400)
        view.addModel(xyz, "xyz", {'vibrate': {'frames':10,'amplitude':1}})
        view.vibrate(10,1.,True,{"radius": 0.15,
                              "radiusRadio":0.8,
                              "mid":30, "color":"#8db600"})
        view.setStyle({'sphere':{'scale':0.30},'stick':{'radius':0.25}})
        view.setBackgroundColor('0xeeeeee')
        view.animate({'loop': 'backAndForth'})
        view.zoomTo()
        
        
        ##############
        
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
        all_frequencies = parse_molden(filename)[0]

        x               = all_frequencies
            # select values

        selected_x = x[mode][0]


        print('wavenumber:', selected_x)

        print('There are {0} normal modes'.format(len(x)))
        
        # Ploting
       
        ax.plot(freq_gn_raman, int_gn_raman, label= "DFT-DB")
        
        
        ax.plot(freq_gn_ir,2-array(int_gn_ir))



        pos_max    = argrelextrema(array(int_gn_ir), np.greater)
        ax.set_xlim(0, freq_gn_ir[pos_max[0][-1]]+50)
        ax.set_xlabel(r"$Frequency [cm^{-1}]$")
        ax.set_ylabel("Intensity")
        
        ax.axvline(selected_x, c="r")
        fig.canvas.draw()
        view.show()
        plt.show()
        

    def show_normal_modes(filename='default.molden_normal_modes'):
        """
        wrapper function that parses the file and initializes the widget.
        """

        
        all_frequencies, coords, normal_modes =  parse_molden(filename=filename)
        _ = interact(draw_normal_mode, coords=fixed(coords), normal_modes=fixed(normal_modes), mode = widgets.Dropdown(
            options=all_frequencies,
            value=0,
            description='Normal mode:',
            style={'description_width': 'initial'}
        ))
        
    return show_normal_modes(filename)


