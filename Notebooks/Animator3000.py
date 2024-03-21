#!/usr/bin/env python
# coding: utf-8
# %%

# %%



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


# %%



def Animator_data(smiles,method):

    r          = requests.get('https://ir.cheminfo.org/v1/ir?smiles={0}&method={1}'.format(smiles, method))
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




