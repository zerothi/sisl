# -*- coding: utf-8 -*-
########################################################################################
# Copyright (c), The Siesta authors. All rights reserved.                       #
#                                                                                      #
# AiiDA-Defects is hosted on GitHub at https://github.com/ConradJohnston/aiida-defects #
# For further information on the license, see the LICENSE.txt file                     #
########################################################################################
from __future__ import absolute_import

import numpy as np


def print_siesta_fdf(path_dir,fdf_name):
    """
    Reading Fdf
    """
    import sisl
   
    Fdf = sisl.io.fdfSileSiesta(path_dir+"/"+fdf_name)

    print ("(1) System label is : {}".format(Fdf.get("systemlabel")))
    print ("(2) Number of Species are : {}".format(Fdf.get("NumberOfSpecies")))
    print ("(3) Total Number of Atoms are : {}".format(Fdf.get("NumberOfAtoms")))
    print ("(4) Chemical Species are : {}".format(Fdf.get("ChemicalSpeciesLabel")))

    return



def read_siesta_fdf_old(path_dir,fdf_name):
    """
    Read The FDF File
    """
    import sisl

    Fdf = sisl.io.fdfSileSiesta(path_dir+"/"+fdf_name)

    #-----------------------------------------------------------------------
    # Species List
    #-----------------------------------------------------------------------
    SpeciesInfoTemp = Fdf.get("ChemicalSpeciesLabel")
    SpeciesInfo = []
    for i in range(len(SpeciesInfoTemp)):
        SpeciesInfo.append(SpeciesInfoTemp[i].split())
    del SpeciesInfoTemp
    SpeciesInfoArray = np.array([])
    for line in SpeciesInfo:
        SpeciesInfoArray = (np.append(SpeciesInfoArray,line))
    SpeciesInfoArray = SpeciesInfoArray.reshape(len(SpeciesInfo),3)
    SpeciesDic={}
    for l in range(len(SpeciesInfo)):
        SpeciesDic[SpeciesInfo[l][2]]=SpeciesInfo[l][0]
    LastSpeciesNumber = len(SpeciesInfo)+1
    Geometry = Fdf.read_geometry()
    # XYZ in Ang Coordinates
    XYZ = Geometry.xyz
    # FracXYZ in Fractional /Internal Coordinates
    FracXYZ = Geometry.fxyz
    # Adding Siesta Z numbers
    AtomSymbol = Geometry.atoms
    Species = np.array([])
    for l in range(len(AtomSymbol)):
        Species = np.append(Species,int(AtomSymbol.specie[l])+1)
    Species = Species.reshape(len(AtomSymbol),1)    
    XYZFull = np.hstack([XYZ,Species])
    # Adding siesta Atom Counter 
    Species = np.array([])
    for l in range(len(AtomSymbol)):
        #print (Species)
        Species = np.append(Species,l+1)
    Species = Species.reshape(len(AtomSymbol),1) 
    XYZFull = np.hstack([XYZFull,Species])
    # Adding siesta Atom Symbols 
    Species = np.array([])
    for l in range(len(AtomSymbol)):
        #print (Species)
        Species = np.append(Species,AtomSymbol[l].symbol)
    Species = Species.reshape(len(AtomSymbol),1)
    XYZFull = np.hstack([XYZFull,Species])    
    
    info_dict = {'Species' : Species,
                 'XYZFull' : XYZFull,
                 'XYZ' : XYZ,
                 'FracXYZ' :FracXYZ
                  }

    return info_dict


def read_siesta_fdf(path_dir,fdf_name,*args):
    """
    Read The FDF File
    """
    import sisl

    print ("Reading Structure from FDF")
    Fdf = sisl.io.fdfSileSiesta(path_dir+"/"+fdf_name)

    Geometry = Fdf.read_geometry()

    info_dict = {'Geometry':Geometry}

    return info_dict

def read_siesta_XV(path_dir,fdf_name):
    """

    """
    import sisl
    print ("Reading Structure from XV")
    Fdf = sisl.io.fdfSileSiesta(path_dir+"/"+fdf_name)
    Geometry = Fdf.read_geometry(output=True,order=['XV'])
    info_dict = {'XV':Geometry}

    return info_dict

def read_siesta_XV_before_relax(path_dir,fdf_name):
    """

    """
    import sisl
    print ("Reading Structure from XV")
    #Fdf = sisl.io.fdfSileSiesta(path_dir+"/"+fdf_name)
    #Geometry = Fdf.read_geometry(output=True,order=['XV'])
    Fdf = sisl.get_sile(path_dir+"/"+fdf_name)
    Geometry = Fdf.read_geometry()
    info_dict = {'XV':Geometry}

    return info_dict



def FixingSislImages(initial,SislStructure,ghost_suffix='ghost',relaxed=False,moving_specie="initial"):
    """
    """
    import sisl
    AtomIndex = np.array([])
    AtomSymbol = np.array([])
    for i in range(SislStructure.na):
        AtomIndex = np.append(AtomIndex,SislStructure.atoms.Z[i])
        AtomSymbol = np.append(AtomSymbol,initial.atoms[i].tag) 
    for i in range(SislStructure.na):

            if i ==0:
                #print(i)
                if AtomIndex[i]< 0:
                    ghost_name = sisl.Atom(-1*AtomIndex[i])
                    Test = sisl.Geometry(xyz=SislStructure.xyz[i],
                            #atoms=sisl.Atom(ghost_name,tag=ghost_name.symbol+"_"+ghost_suffix), Bug!
                            atoms=sisl.Atom(int(AtomIndex[i]),tag=ghost_name.symbol+"_"+ghost_suffix),
                            sc=SislStructure.cell)
                else:
                    Test = sisl.Geometry(xyz=SislStructure.xyz[i],
                            atoms=sisl.Atom(AtomIndex[i],tag=AtomSymbol[i]),
                            sc=SislStructure.cell
                               )
            #if i>0 and i<= SislStructure.na-2:
            if 0<i<= SislStructure.na-2:
                #print(i)
                if AtomIndex[i]< 0:
                    ghost_name = sisl.Atom(-1*AtomIndex[i])
                    Test += sisl.Geometry(xyz=SislStructure.xyz[i],
                            #atoms=sisl.Atom(ghost_name,tag=ghost_name.symbol+"_"+ghost_suffix), Bug!!
                            atoms=sisl.Atom(int(AtomIndex[i]),tag=ghost_name.symbol+"_"+ghost_suffix),
                            sc=SislStructure.cell)
                else:
                    Test += sisl.Geometry(xyz=SislStructure.xyz[i],
                            atoms=sisl.Atom(AtomIndex[i],tag=AtomSymbol[i]),
                            sc=SislStructure.cell
                                )
            if SislStructure.na-2 < i < SislStructure.na:
                #print(i)
                if relaxed :
                    Test += sisl.Geometry(xyz=SislStructure.xyz[i],
                            #atoms=sisl.Atom(AtomIndex[i],tag=AtomSymbol[i]+suffix),
                            atoms=sisl.Atom(AtomIndex[i],tag=AtomSymbol[i]),
                            sc=SislStructure.cell
                                )
                else:
                    Test += sisl.Geometry(xyz=SislStructure.xyz[i],
                            #atoms=sisl.Atom(AtomIndex[i],tag=AtomSymbol[i]+suffix),
                            atoms=sisl.Atom(AtomIndex[i],tag=AtomSymbol[i]+"_"+moving_specie),
                            sc=SislStructure.cell
                                )
    return Test

