#!/usr/bin/env python
# coding: utf-8

# In[ ]:

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

plt.rcParams.update({'font.size': 14})
# %%

def Animator_data(smiles,method):
    smiles_xTB = smiles.replace("#","%23")
    r          = requests.get('https://ir.cheminfo.org/v1/ir?smiles={0}&method={1}'.format(smiles_xTB, method))
    data_bytes = r.content
    data_dic   = json.loads(data_bytes.decode('utf-8'))

    norm_int   = []
    norm_int_R = []

    wavenumb   = array(data_dic["wavenumbers"])

    modes = data_dic["modes"]
    vib_modes=[]
    for i in range(0,len(modes)):
        if modes[i]['modeType']=='vibration':
            vib_modes.append(modes[i])
    frequency = []
    for i in range(0,len(vib_modes)):
        frequency.append(float(vib_modes[i]["wavenumber"]))
        
    frequency_c = [ '%.2f' % elem for elem in frequency ]

    a        = vib_modes[0]["displacements"]
    ar_s     = a.split("\n")
    modes_al = ar_s[2:-1]

    modes_al_c = []
    for i in modes_al:
        modes_al_c.append((" ".join(i.split())).split())

    #Writing the input
    a        = vib_modes[0]["displacements"]
    ar_s     = a.split("\n")
    modes_al = ar_s[2:-1]

    modes_al_c = []
    for i in modes_al:
        modes_al_c.append((" ".join(i.split())).split())

    modes_al_c
    with open('readme.txt', 'w') as f:

        f.write("[Molden Format]")
        f.write('\n')
        f.write('\n')

        f.write("[FREQ]")
        f.write('\n')

        for line in frequency_c:
            f.write((line))
            f.write('\n')

        f.write('\n')
        f.write("[FR-COORD]")
        f.write('\n')
        for i in range(0,len(modes_al_c)):
            for k in range(0,4):
                f.write(modes_al_c[i][k])
                f.write('       ')
            f.write('\n')
        f.write('\n')

        f.write("[FR-NORM-COORD]")
        f.write('\n')
        for vib in range(0,len(vib_modes)):
            a        = vib_modes[vib]["displacements"]
            ar_s     = a.split("\n")
            modes_al = ar_s[2:-1]

            modes_al_c = []
            for i in modes_al:
                modes_al_c.append((" ".join(i.split())).split())


            f.write("vibration {0}".format(vib+1))
            f.write('\n')
            for i in range(0,len(modes_al_c)):
                f.write('           ')
                for k in range(4,7):

                    f.write(modes_al_c[i][k]+"00000")
                    f.write('        ')
                f.write('\n')

wavenumbers=linspace(0,4000,10001)


### Gausian function to broaden peaks
def g(wavenumb_sweep, intensity_max, wavenumber_max, σ):
    G = intensity_max / (σ *sqrt(2 * pi)) * exp(-(wavenumb_sweep-wavenumber_max)**2 / (2*σ**2))
    new_y=array(G)  
    return new_y
       
### Getting IR from xTB GFN2xTB, GFN1xTB and GFNFF
def IR_xTB(smiles, method, σ,σ_R):
    
    smiles_xTB = smiles.replace("#","%23")

    r          = requests.get('https://ir.cheminfo.org/v1/ir?smiles={0}&method={1}'.format(smiles_xTB, method))
    print(r)
    data_bytes = r.content
    
    
    data_dic   = json.loads(data_bytes.decode('utf-8'))
    
    norm_int   = []
    norm_int_R = []
    
    wavenumb   = array(data_dic["wavenumbers"])
    
    max_int    = max(data_dic["intensities"])
    max_int_Ram= max(data_dic["ramanIntensities"])
    
    for i in data_dic["intensities"]:
        norm_int.append(i/max_int)
    for i in data_dic["ramanIntensities"]:
        norm_int_R.append(i/max_int_Ram)
        

    ### Gausian function to broaden peaks
    pos_max    = argrelextrema(array(norm_int), np.greater)
    x          = wavenumb
    all_curve = 0
    
    for i in pos_max[0]:
        all_curve += g(x, norm_int[i], wavenumb[i],σ)      
    broad_int = all_curve   
    
    #Raman
    pos_max_R    = argrelextrema(array(norm_int_R), np.greater)
    x_R          = wavenumb
    all_curve_R = 0
    
    for i in pos_max_R[0]:
        all_curve_R += g(x_R, norm_int_R[i], wavenumb[i],σ_R)      
    broad_int_R = all_curve_R  
        
    ### Normalization
    
    max_y = max(broad_int)
    int_norm = []
    
    for i in broad_int:
        int_norm.append((i/max_y))
        
    ### Normalization Raman
    
    max_y_R = max(broad_int_R)
    int_norm_R = []
    
    for i in broad_int_R:
        int_norm_R.append((i/max_y_R))
       
    #Transmitance
    trans = []
    for i in int_norm: 
        trans.append(1-i)
    
    # DIPOLE MOMENT
   
    wavenumb   = array(data_dic["wavenumbers"])
    
    # Import normal modes
    modes = data_dic["modes"]
    vib_modes=[]
    for i in range(0,len(modes)):
        if modes[i]['modeType']=='vibration':
            vib_modes.append(modes[i])
            
    # Import frequencies
    frequency=[]
    for i in range(0,len(vib_modes)):
        frequency.append(vib_modes[i]["wavenumber"])
        
    # Import normal modes coordinates
    a        = vib_modes[0]["displacements"]
    ar_s     = a.split("\n")
    modes_al = ar_s[2:-1]

    modes_al_c = []
    for i in modes_al:
        modes_al_c.append((" ".join(i.split())).split())

    print("There are {0} normal modes".format(len(vib_modes)))
   
    elements = []
    for i in modes_al_c:
        elements.append(i[0])
            
    print(elements)  
    # Define the function to calculate the dipole moment for a configuration
    def calculate_dipole_moment(coordinates, charges, masses):
        # Calculate the center of mass
        total_mass = jnp.sum(masses)
        center_of_mass = jnp.sum(coordinates * masses[:, None], axis=0) / total_mass

        # Calculate the dipole moment
        dipole_moment = jnp.zeros(3)

        for charge, (x, y, z) in zip(charges, coordinates):
            dipole_moment += charge * jnp.array([x, y, z]) - charge * center_of_mass

        return dipole_moment

    # Define the function for linear interpolation of configurations
    def interpolate_configurations(initial_coords, final_coords, num_steps):
        steps = jnp.linspace(0.0, 1.0, num_steps)[:, None, None]
        return initial_coords + steps * (final_coords - initial_coords)


    def calculate_partial_charges(molecule_smiles):
        # Create an RDKit molecule object
        mol = Chem.MolFromSmiles(molecule_smiles)

        # Add hydrogens and generate 3D coordinates
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, AllChem.ETKDG())

        # Compute Gasteiger partial charges
        AllChem.ComputeGasteigerCharges(mol)

        # Get the computed partial charges for each atom
        partial_charges = [atom.GetDoubleProp("_GasteigerCharge") for atom in mol.GetAtoms()]
        atom_masses = []
        for atom in mol.GetAtoms():
            atom_mass = atom.GetMass()
            atom_masses.append(atom_mass)

        return partial_charges, atom_masses

    partial_charges, atom_masses = calculate_partial_charges(smiles)
    
    charges = jnp.array(partial_charges)
    masses  = jnp.array(atom_masses)
    
    print("Tha charges are {0} with a mass of {1}".format(charges, masses))

    ########## ---------- Writing the initial positions --------- 
    μ_all= []
    α_all= []
    all_pol_mode = []
    for mode in range(0,len(vib_modes)):
        μ_i_f = []
        a        = vib_modes[mode]["displacements"]
        ar_s     = a.split("\n")
        modes_al = ar_s[2:-1]

        modes_al_c = []
        for i in modes_al:
            modes_al_c.append((" ".join(i.split())).split())

        num_modes = []
        elements = []
        for i in modes_al_c:
            atom      = []
            for k in range(1,4):
                atom.append(float(i[k])) 
            elements.append(i[0])
            num_modes.append(atom)

        coordinates = []
        for i in num_modes:
            coordinates.append(tuple(i))
        initial_coordinates = jnp.array(num_modes)
#         print(initial_coordiantes)
        

        #### Final positions  
        vib_modes_1 = []

        for i in modes_al_c:
            atom      = []
            for k in range(4,7):
                atom.append(float(i[k])) 
            vib_modes_1.append(atom)

        coordinates_vib = []
        for i in vib_modes_1:
            coordinates_vib.append(tuple(i))
        final_coordinates = jnp.array(vib_modes_1)
        # DIPOLE MOMENT CALCULATION 
        

        # Compute 100 intermediate configurations using linear interpolation for each coordinate
        num_interpolations = 100
        interpolated_coordinates = interpolate_configurations((initial_coordinates)[None, :],\
                                                              (initial_coordinates+final_coordinates)[None, :],\
                                                              num_interpolations)

        # Calculate the dipole moment magnitudes for each configuration
        dipole_moment_magnitudes = vmap(calculate_dipole_moment, (0, None, None))(interpolated_coordinates, charges, masses)

        # Print the list of dipole moment magnitudes
        # print(dipole_moment_magnitudes)
        news=[]
        for i in dipole_moment_magnitudes:
            news.append(jnp.linalg.norm(i))
        μ_all.append(news)

    #   POLARIZABILITY
        # Compute 10 intermediate configurations using linear interpolation for each coordinate
        num_interpolations = 10
        interpolated_coordinates = interpolate_configurations((initial_coordinates-final_coordinates)[None, :],\
                                                                (initial_coordinates+final_coordinates)[None, :], \
                                                                num_interpolations)
        index_IC = 0
        # Creating documents for polarizability 
        pol_mode =[]
        files_gen=[]
        
        for config in interpolated_coordinates:
            index_IC+=1
            xyz_file = "mode_{0}_{1}_{2}_{3}.xyz".format(mode,smiles,method, index_IC)
            myfile = open(xyz_file,"w")
            myfile.write("{0}".format(len(elements)))
            myfile.write('\n')
            myfile.write('\n')
            for i in range(0,len(elements)):
                myfile.write("{0} {1} {2} {3}".format(elements[i],\
                                                          config[i][0],\
                                                          config[i][1],\
                                                         config[i][2]))
                myfile.write('\n') 
            myfile.close()
            files_gen.append(xyz_file)
            
            command = ["kallisto", "alp", "--molecular", xyz_file]
             # Execute the command
            output  = subprocess.check_output(command, text=True)
            pol_mode.append(float(output))


        all_pol_mode.append(pol_mode) 
    return  int_norm, trans, int_norm_R,norm_int, μ_all, all_pol_mode

def IR_axo( filename,Compound_wn, Compound_int,smiles_code, method, sigma, sigma_raman,μ_all, all_pol_mode ):
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

    def draw_normal_mode( mode=0, coords=None, normal_modes=None):

        xyz =f"{len(coords)}\n\n"
        for i in range(len(coords)):
            atom_coords = [float(m) for m in  coords[i][8:].split('       ')]
            mode_coords = [float(m) for m in  normal_modes[mode][i][9:].split('       ')]
            xyz+=f"{coords[i][0:4]} {atom_coords[0]} {atom_coords[1]} {atom_coords[2]} {mode_coords[0]} {mode_coords[1]} {mode_coords[2]} \n"
        view = py3Dmol.view(width=300, height=300)
        view.addModel(xyz, "xyz", {'vibrate': {'frames':10,'amplitude':1}})
        view.vibrate(10,1.,True,{"radius": 0.15,
                          "radiusRadio":0.8,
                          "mid":30, "color":"#8db600"})
        view.setStyle({'sphere':{'scale':0.30},'stick':{}})
        view.setBackgroundColor('0xeeeeee')
        view.animate({'loop': 'backAndForth'})
        view.zoomTo()


        wavenumbers                = linspace(0,4000,10001) 

        fig = plt.figure(figsize=(14, 7))
        gs = GridSpec(nrows=2, ncols=2)


        
        all_frequencies = parse_molden(filename)[0]

        x               = all_frequencies
            # select values

        selected_x = x[mode][0]


        print('wavenumber:', selected_x)

        print('There are {0} normal modes active in IR'.format(len(x)))

        ax0=fig.add_subplot(gs[:, 0])
        ax0.plot(Compound_wn, 0.5*array(Compound_int[1])+0.5, label="Theo. {0}- IR".format(method))
        ax0.plot(Compound_wn, 0.5*array(Compound_int[2]), label="Theo. {0} -Raman".format(method))
        ax0.set_ylim(0,1.6)
        ax0.get_yaxis().set_visible(False)
        ax0.set_ylabel("Intensity")
        ax0.set_xlabel("wavenumber $[cm^{-1}]$")
        ax0.invert_xaxis()
        ax0.legend(loc=1)
        mol = Chem.MolFromSmiles(smiles_code)
        size =150

        im = Chem.Draw.MolToImage(mol, size=(size,size))
    #     ax = plt.axes([0.6, 0.47, 0.38, 0.38], frameon=True)

        imagebox = OffsetImage(im, zoom = 0.8)
        ab = AnnotationBbox(imagebox, (3200, 1.3), frameon = False)
        fig.canvas.draw()
        ax0.add_artist(ab)

        ax0.axvline(selected_x, c="r")
        
        ax1 = fig.add_subplot(gs[0, 1])
        ax1.plot(μ_all[mode], label=r"$\nu= {0} \ cm-1$".format(round(x[mode][0],2) ))
        ax1.set_ylim(-0.01,max(μ_all[mode])+0.01)
        ax1.set_ylabel("$|\mu|$")
        ax1.legend()
        fig.canvas.draw()

        
        
        ax2 = fig.add_subplot(gs[1, 1])
        ax2.plot(all_pol_mode[mode])
        # ax1.set_ylim(-0.01,max(μ_all[mode])+0.01)
        ax2.set_ylabel("$|α |$")
        ax2.set_xlabel("Normal coordinate")

        fig.canvas.draw()
        view.show()
        plt.show()


    def show_normal_modes(filename):
        """
        wrapper function that parses the file and initializes the widget.
        """
        all_frequencies, coords, normal_modes =  parse_molden(filename=filename)

        dropdown = widgets.Dropdown(
            options=all_frequencies,
            value=0,
            description='Normal mode:',
            style={'description_width': 'initial'}
        )

        _ = interact(draw_normal_mode, coords=fixed(coords), normal_modes=fixed(normal_modes), mode = dropdown )
    return show_normal_modes(filename)



# %%

