# -*- coding: utf-8 -*-
########################################################################################
# Copyright (c), The Siesta authors. All rights reserved.                       #
#                                                                                      #
# AiiDA-Defects is hosted on GitHub at https://github.com/ConradJohnston/aiida-defects #
# For further information on the license, see the LICENSE.txt file                     #
########################################################################################
from __future__ import absolute_import

import numpy as np
def is_frac_or_cart_or_index(Position):
    """
    """
    a_dummy = np.array([])
    if type(Position) == type(a_dummy) :
        if Position.shape == (3,):
            print("It's either Cart or Frac")
            if Position.max()>1.0:
                print("The Atom Positions given in Cartesian (Ang)  " )
                out = 'cartesian'
            else:
                print("The Atom Positions given in Internal (fractional)  " )
                out = 'frac'
        else:
            print("Something is Wrong in InitialAtomPosition or FinalAtomPosition")
    elif type(Position) == type(1):
        print ("It's index")
        out = 'index'
    else:
        print("Something is Wrong in InitialAtomPosition")

    return out



def AtomIndex(Positions,AtomPosition,rtol,atol):
    """
     Positions = FracXYZ
     AtomPosition = InitialAtomPosition
     rtol 
     atol
     
     Read position and Specific Postions coordinates to return the index number of array
     
    """

    for i in range(len(Positions)):
        if np.isclose(float(Positions[i][0]),AtomPosition[0],rtol,atol) and np.isclose(float(Positions[i][1]),AtomPosition[1],rtol,atol) and np.isclose(float(Positions[i][2]),AtomPosition[2],rtol,atol):
            #print ("Index and Atomic Position is ",i,Positions[i])
            index=i
    try:
        return index
    except:
          print(" Couldn't Find the Atom")
    
