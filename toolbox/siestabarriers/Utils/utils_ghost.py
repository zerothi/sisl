# -*- coding: utf-8 -*-
########################################################################################
# Copyright (c), The Siesta authors. All rights reserved.                       #
#                                                                                      #
# AiiDA-Defects is hosted on GitHub at https://github.com/ConradJohnston/aiida-defects #
# For further information on the license, see the LICENSE.txt file                     #
########################################################################################
from __future__ import absolute_import

import numpy as np


def ghost(frac_or_cart,):
    """

    """

    print ("------------------------------------")
    print (" ...... Processing Ghost Atom ..... ")
    print ("------------------------------------")
    # Checking how many type of Ghost is there
    GhostInitialASEXYZ = XYZFull
    GhostFinalASEXYZ = XYZFull
    if frac_or_cart == True:
        GhostASEXYZ = GhostInitialASEXYZ[AtomIndex(XYZ,InitialAtomPosition,rtol,atol)]
        GhostASEXYZ = np.vstack([GhostASEXYZ,GhostFinalASEXYZ[AtomIndex(XYZ,FinalAtomPosition,rtol,atol)]])
    else:
        GhostASEXYZ = GhostInitialASEXYZ[AtomIndex(FracXYZ,InitialAtomPosition,rtol,atol)]
        GhostASEXYZ = np.vstack([GhostASEXYZ,GhostFinalASEXYZ[AtomIndex(FracXYZ,FinalAtomPosition,rtol,atol)]])
    #--------------------------------------------
    # If Exchange-vacancy  
    #--------------------------------------------
    if (GhostASEXYZ[0][5] == GhostASEXYZ[1][5]):
        print("The Ghosts Are Same Species")
        SpeciesDic[GhostASEXYZ[0][5]+"_ghost"] = LastSpeciesNumber
        for l in range(len(GhostASEXYZ)):
            print(GhostASEXYZ[l])
        # Saving Keys and Values for Later use
        SpeciesDic_Key = []
        SpeciesDic_Value = []
        for key, value in SpeciesDic.items():
            SpeciesDic_Key.append(key)
            SpeciesDic_Value.append(value)
        #len(SpeciesDic_Value)
        for i in range(len(SpeciesInfoArray)):
            if np.isin(GhostASEXYZ[0][5],SpeciesInfoArray[i][2]):
            # DEBUG if np.isin('La',SpeciesInfoArray[i][2]):
                ghost_i = i
                print ("The Ghost Detail :"+ str(SpeciesInfoArray[ghost_i]))

        GhostSpeciesTemp = np.array([])
        TempA = np.array([[len(SpeciesDic_Value),'-'+SpeciesInfoArray[ghost_i][1],GhostASEXYZ[0][5]+"_ghost"]])
        SpeciesInfoArray = np.vstack([SpeciesInfoArray,TempA])

    else:
        print("The Ghosts Are Different Species")
        for l in range(len(GhostASEXYZ)):
            SpeciesDic[GhostASEXYZ[l][5]+"_ghost"]=LastSpeciesNumber+l
        for l in range(len(GhostASEXYZ)):
            print(GhostASEXYZ[l])
        # Saving Keys and Values for Later use
        SpeciesDic_Key = []
        SpeciesDic_Value = []
        for key, value in SpeciesDic.items():
            SpeciesDic_Key.append(key)
            SpeciesDic_Value.append(value)
        #for l in range(len(GhostASEXYZ)):
        #    print(LastSpeciesNumber+l)

        TempA = np.array([])
        GhostSpeciesTemp = np.array([])
        for i in range(len(GhostASEXYZ)):
            #print (GhostASEXYZ[i][5])
            #print (i)
            TempA = np.append(TempA,np.array([[len(SpeciesDic_Value)+i-1,'-'+SpeciesInfoArray[i][1],GhostASEXYZ[i][5]+"_ghost"]]))
        TempA = TempA.reshape(2,3)
        SpeciesInfoArray = np.vstack([SpeciesInfoArray,TempA])
        print("DONE!")

