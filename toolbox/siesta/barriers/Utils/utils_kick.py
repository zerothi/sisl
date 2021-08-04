# -*- coding: utf-8 -*-
########################################################################################
# Copyright (c), The Siesta authors. All rights reserved.                       #
#                                                                                      #
# AiiDA-Defects is hosted on GitHub at https://github.com/ConradJohnston/aiida-defects #
# For further information on the license, see the LICENSE.txt file                     #
########################################################################################
from __future__ import absolute_import

import numpy as np

#========================================================================================
#
#
#
#========================================================================================

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


def ASE2Siesta(A):
    """
    Getting ASE Atom Object and Converted to sisl
    """
    import sisl
    geometry = sisl.Geometry.fromASE(A)
    return geometry


def pre_prepare_sisl (frac_or_cart_or_index,Initial_Geom,InitialAtomPosition,FinalAtomPosition,KickedAtomPosition,rtol=1e-2,atol=1e-2):

    """
    
    """
    import sisl
    from sisl import Atom
    if frac_or_cart_or_index == 'cartesian':
        print ("Cartesian ...")
        #Initial_Geom = Fdf.read_geometry()
        XYZ = Initial_Geom.xyz
        InitialASEXYZ = Initial_Geom
        FinalASEXYZ = Initial_Geom
        print ("Removing Vacancies Ang/Bohr")
        print ("Removing Index for Initial Atom:{}".format(AtomIndex(XYZ,InitialAtomPosition,rtol,atol)))
        print ("Removing Index for Final Atom:{}".format(AtomIndex(XYZ,FinalAtomPosition,rtol,atol)))
        
        #TraceAtom_A_Initial = sisl.Geometry(Initial_Geom.xyz[AtomIndex(XYZ,InitialAtomPosition,rtol,atol)],
        #                              atoms= Initial_Geom.atoms.Z[AtomIndex(XYZ,InitialAtomPosition,rtol,atol)])
        #TraceAtom_A_Final = sisl.Geometry(Initial_Geom.xyz[AtomIndex(XYZ,FinalAtomPosition,rtol,atol)],
        #                            atoms=  Initial_Geom.atoms.Z[AtomIndex(XYZ,InitialAtomPosition,rtol,atol)])       
        #TraceAtom_B_Initial = sisl.Geometry(Initial_Geom.xyz[AtomIndex(XYZ,FinalAtomPosition,rtol,atol)],
        #                            atoms=  Initial_Geom.atoms.Z[AtomIndex(XYZ,FinalAtomPosition,rtol,atol)])
        #TraceAtom_B_Kicked = sisl.Geometry(KickedAtomPosition,
        #                            atoms=  Initial_Geom.atoms.Z[AtomIndex(XYZ,FinalAtomPosition,rtol,atol)])
        #TraceAtom_B_Kicked = sisl.Geometry(Initial_Geom.xyz[AtomIndex(XYZ,KickedAtomPosition,rtol,atol)],
        #                            atoms=  Initial_Geom.atoms.Z[AtomIndex(XYZ,FinalAtomPosition,rtol,atol)])


        # Fixing tags
        TraceAtom_A_Initial_Info = sisl.Atom(Initial_Geom.atoms.Z[AtomIndex(XYZ,InitialAtomPosition,rtol,atol)])
        TraceAtom_A_Initial = sisl.Geometry(xyz=Initial_Geom.xyz[AtomIndex(XYZ,InitialAtomPosition,rtol,atol)],
                                           atoms= sisl.Atom(TraceAtom_A_Initial_Info.Z,tag=TraceAtom_A_Initial_Info.symbol                                                  ))
         
        TraceAtom_A_Final_Info  = sisl.Atom(Initial_Geom.atoms.Z[AtomIndex(XYZ,InitialAtomPosition,rtol,atol)])
        TraceAtom_A_Final = sisl.Geometry(Initial_Geom.xyz[AtomIndex(XYZ,FinalAtomPosition,rtol,atol)],
                                    atoms= sisl.Atom(TraceAtom_A_Final_Info.Z,tag=TraceAtom_A_Final_Info.symbol))       
        
        TraceAtom_B_Initial_Info = sisl.Atom(Initial_Geom.atoms.Z[AtomIndex(XYZ,FinalAtomPosition,rtol,atol)] )
        TraceAtom_B_Initial = sisl.Geometry(Initial_Geom.xyz[AtomIndex(XYZ,FinalAtomPosition,rtol,atol)],
                                    atoms= sisl.Atom (TraceAtom_B_Initial_Info.Z,tag=TraceAtom_B_Initial_Info.symbol))
        
        TraceAtom_B_Kicked_Info = sisl.Atom(Initial_Geom.atoms.Z[AtomIndex(XYZ,FinalAtomPosition,rtol,atol)] )
        TraceAtom_B_Kicked = sisl.Geometry(KickedAtomPosition,
                                    atoms= sisl.Atom(TraceAtom_B_Kicked_Info.Z,tag=TraceAtom_B_Kicked_Info.symbol))
        

        InitialASEXYZ = InitialASEXYZ.remove(AtomIndex(XYZ,InitialAtomPosition,rtol,atol))
        FinalASEXYZ = FinalASEXYZ.remove(AtomIndex(XYZ,FinalAtomPosition,rtol,atol))
        if AtomIndex(XYZ,FinalAtomPosition,rtol,atol) > AtomIndex(XYZ,InitialAtomPosition,rtol,atol):
            print ("Order : Final Atom Position > Initial Atom Position")
            InitialASEXYZ = InitialASEXYZ.remove(AtomIndex(XYZ,FinalAtomPosition,rtol,atol)-1)
            FinalASEXYZ = FinalASEXYZ.remove(AtomIndex(XYZ,InitialAtomPosition,rtol,atol))
        if AtomIndex(XYZ,FinalAtomPosition,rtol,atol) < AtomIndex(XYZ,InitialAtomPosition,rtol,atol):
            print ("Order : Initial Atom Position > Final Atom Position")
            InitialASEXYZ = InitialASEXYZ.remove(AtomIndex(XYZ,FinalAtomPosition,rtol,atol))
            FinalASEXYZ = FinalASEXYZ.remove(AtomIndex(XYZ,InitialAtomPosition,rtol,atol)-1)   
    elif frac_or_cart_or_index  == 'frac':
        #print ("Removing Vacancies Fractional NOT Implemented")
        print ("Fractional ...")
        Frac = Initial_Geom.fxyz
        InitialASEXYZ = Initial_Geom
        FinalASEXYZ = Initial_Geom
        print ("Removing Vacancies Ang/Bohr")
        print ("Removing Index for Initial Atom:{}".format(AtomIndex(Frac,InitialAtomPosition,rtol,atol)))
        print ("Removing Index for Final Atom:{}".format(AtomIndex(Frac,FinalAtomPosition,rtol,atol)))
        trace_atom_initial = sisl.Geometry(Initial_Geom.xyz[AtomIndex(Frac,InitialAtomPosition,rtol,atol)],
                                      atoms= Initial_Geom.atoms.Z[AtomIndex(Frac,InitialAtomPosition,rtol,atol)])
        trace_atom_final = sisl.Geometry(Initial_Geom.xyz[AtomIndex(Frac,FinalAtomPosition,rtol,atol)],
                                        atoms= Initial_Geom.atoms.Z[AtomIndex(Frac,FinalAtomPosition,rtol,atol)])    

        InitialASEXYZ = InitialASEXYZ.remove(AtomIndex(Frac,InitialAtomPosition,rtol,atol))
        FinalASEXYZ = FinalASEXYZ.remove(AtomIndex(Frac,FinalAtomPosition,rtol,atol))
        if AtomIndex(Frac,FinalAtomPosition,rtol,atol) > AtomIndex(Frac,InitialAtomPosition,rtol,atol):
            print ("Order : Final Atom Position > Initial Atom Position")
            InitialASEXYZ = InitialASEXYZ.remove(AtomIndex(Frac,FinalAtomPosition,rtol,atol)-1)
            FinalASEXYZ = FinalASEXYZ.remove(AtomIndex(Frac,InitialAtomPosition,rtol,atol))
        if AtomIndex(Frac,FinalAtomPosition,rtol,atol) < AtomIndex(Frac,InitialAtomPosition,rtol,atol):
            print ("Order : Initial Atom Position > Final Atom Position")
            InitialASEXYZ = InitialASEXYZ.remove(AtomIndex(Frac,FinalAtomPosition,rtol,atol))
            FinalASEXYZ = FinalASEXYZ.remove(AtomIndex(Frac,InitialAtomPosition,rtol,atol)-1)
    else:
        print('index')
        #Frac = Initial_Geom.fxyz
        InitialASEXYZ = Initial_Geom
        FinalASEXYZ = Initial_Geom
        print ("Removing Vacancies Ang/Bohr")
        print ("Removing Index for Initial Atom:{}".format(InitialAtomPosition-1))
        print ("Removing Index for Final Atom:{}".format(FinalAtomPosition-1))
        
        #trace_atom_A_initial = sisl.Geometry(Initial_Geom.xyz[InitialAtomPosition-1],
        #                                   atoms= Initial_Geom.atoms.Z[InitialAtomPosition-1])
        #trace_atom_B_final = sisl.Geometry(Initial_Geom.xyz[FinalAtomPosition-1],
        #                                 atoms=  Initial_Geom.atoms.Z[FinalAtomPosition-1])       
        trace_atom_A_initial = sisl.Geometry(Initial_Geom.xyz[InitialAtomPosition-1],
                                      atoms= Initial_Geom.atoms.Z[InitialAtomPosition-1])
        trace_atom_A_final = sisl.Geometry(Initial_Geom.xyz[FinalAtomPosition-1],
                                      atoms= Initial_Geom.atoms.Z[InitialAtomPosition-1])
        trace_atom_B_initial = sisl.Geometry(Initial_Geom.xyz[FinalAtomPosition-1],
                                      atoms= Initial_Geom.atoms.Z[FinalAtomPosition-1])
        trace_atom_B_final = sisl.Geometry(Initial_Geom.xyz[InitialAtomPosition-1],
                                      atoms= Initial_Geom.atoms.Z[FinalAtomPosition-1])


        InitialASEXYZ = InitialASEXYZ.remove(InitialAtomPosition-1)
        FinalASEXYZ = FinalASEXYZ.remove(FinalAtomPosition-1)
        if (FinalAtomPosition-1) > (InitialAtomPosition-1):
            print ("Order : Final Atom Position > Initial Atom Position")
            InitialASEXYZ = InitialASEXYZ.remove(FinalAtomPosition-2)
            FinalASEXYZ = FinalASEXYZ.remove(InitialAtomPosition-1)
        if (FinalAtomPosition-1) < (InitialAtomPosition-1):
            print ("Order : Initial Atom Position > Final Atom Position")
            InitialASEXYZ = InitialASEXYZ.remove(FinalAtomPosition-1)
            FinalASEXYZ = FinalASEXYZ.remove(InitialAtomPosition-2)   

    
    

    InitialASEXYZ = InitialASEXYZ.add(TraceAtom_A_Initial)
    FinalASEXYZ = FinalASEXYZ.add(TraceAtom_A_Final)

    info_sisl = {'initial' : InitialASEXYZ,
                 'final' : FinalASEXYZ,
                 'trace_atom_A_initial' : TraceAtom_A_Initial,
                 'trace_atom_A_final' : TraceAtom_A_Final,
                 'trace_atom_B_initial' : TraceAtom_B_Initial,
                 'trace_atom_B_kicked' : TraceAtom_B_Kicked,
                 }


    return info_sisl


def pre_prepare_ase_after_relax(initial_XV,final_XV):
    """
    """
    import sisl
    trace_atom_A_initial = sisl.Geometry(initial_XV.xyz[-2],
                                      atoms = initial_XV.atoms.Z[-2])
    trace_atom_A_final = sisl.Geometry( final_XV.xyz[-2],
                                    atoms =  final_XV.atoms.Z[-2])
    trace_atom_B_initial = sisl.Geometry( initial_XV.xyz[-1],
                                    atoms=  initial_XV.atoms.Z[-1])
    trace_atom_B_kicked = sisl.Geometry(final_XV.xyz[-1],
                                    atoms=  final_XV.atoms.Z[-1])

    #trace_atom_A_initial = sisl.Geometry(initial_XV.xyz[-2],
    #                                  atoms =  initial_XV.atoms.Z[-2])
    #trace_atom_B_initial = sisl.Geometry(initial_XV.xyz[-1],
    #                                atoms = initial_XV.atoms.Z[-1])
    initial_XV = initial_XV.remove([-1])
    final_XV = final_XV.remove([-1])
       
    info_sisl = {'initial' : initial_XV,
                 'final' : final_XV,
                 'trace_atom_A_initial' : trace_atom_A_initial,
                 'trace_atom_A_final' : trace_atom_A_final,
                 'trace_atom_B_initial' : trace_atom_B_initial,
                 'trace_atom_B_kicked' : trace_atom_B_kicked,
                 }

    return info_sisl





#def moving_species(images):
#    """
#    """



#def ASEInitilizer(Initial,InitialPositions,Relaxed):
#    """
#
#    """
#    import ase 
#    #initializing Image
#    initialized=ase.Atoms()
#    if Relaxed == False:
#        print ("Initializing for ASE Before Relaxation")
#        for i in range(len(Initial)):
#            #print ("initialize =",i)
#            initialized+=ase.Atoms(Initial[i][5],positions=[(float(InitialPositions[i][0]),
#                                                             float(InitialPositions[i][1]),
#                                                             float(InitialPositions[i][2]))
#                                                             ])
#    if Relaxed == True :
#        #initializing Initial Image
#        initial=ase.Atoms()
#        print ("Initializing for ASE After Relaxation")
#        for i in range(len(Initial)):
#            #print ("initialize =",i)
#            initialized+=ase.Atoms(Initial[i][5],positions=[(InitialPositions.xyz[i][0],
#                                                             InitialPositions.xyz[i][1],
#                                                             InitialPositions.xyz[i][2])])
#
#    return(initialized)

#def prepare_ase_old(frac_or_cart,XYZ,XYZFull,FracXYZ, InitialAtomPosition , FinalAtomPosition ,rtol ,atol):
#    """
#    """
#    if frac_or_cart  == True:
#        print ("Cartesian ...")
#        InitialASEXYZ = XYZFull
#        FinalASEXYZ = XYZFull
#        print ("Removing Vacancies Ang/Bohr")
#        print ("Removing Index for Initial Atom:{}".format(AtomIndex(XYZ,InitialAtomPosition,rtol,atol)))
#        print ("Removing Index for Final Atom:{}".format(AtomIndex(XYZ,FinalAtomPosition,rtol,atol)))
#        InitialASEXYZ = np.delete(InitialASEXYZ,AtomIndex(XYZ,InitialAtomPosition,rtol,atol),0)
#        FinalASEXYZ = np.delete(FinalASEXYZ,AtomIndex(XYZ,FinalAtomPosition,rtol,atol),0)
#        if AtomIndex(XYZFull,FinalAtomPosition,rtol,atol) > AtomIndex(XYZFull,InitialAtomPosition,rtol,atol):
#            print ("Final Atom Position > Initial Atom Position")
#            InitialASEXYZ = np.delete(InitialASEXYZ,AtomIndex(XYZFull,FinalAtomPosition,rtol,atol)-1,0)
#            FinalASEXYZ = np.delete(FinalASEXYZ,AtomIndex(XYZFull,InitialAtomPosition,rtol,atol),0)
#        if AtomIndex(XYZFull,FinalAtomPosition,rtol,atol) < AtomIndex(XYZFull,InitialAtomPosition,rtol,atol):
#            print ("Initial Atom Position > Final Atom Position")
#            InitialASEXYZ = np.delete(InitialASEXYZ,AtomIndex(XYZFull,FinalAtomPosition,rtol,atol),0)
#            FinalASEXYZ = np.delete(FinalASEXYZ,AtomIndex(XYZFull,InitialAtomPosition,rtol,atol)-1,0)
#    else:
#        #print ("Removing Vacancies Fractional NOT Implemented")
#        print ("Fractional ...")
#        InitialASEXYZ = XYZFull
#        FinalASEXYZ = XYZFull
#        print ("Removing Vacancies Ang/Bohr")
#        print ("Removing Index for Initial Atom:{}".format(AtomIndex(FracXYZ,InitialAtomPosition,rtol,atol)))
#        print ("Removing Index for Final Atom:{}".format(AtomIndex(FracXYZ,FinalAtomPosition,rtol,atol)))
#        InitialASEXYZ = np.delete(InitialASEXYZ,AtomIndex(FracXYZ,InitialAtomPosition,rtol,atol),0)
#        FinalASEXYZ = np.delete(FinalASEXYZ,AtomIndex(FracXYZ,FinalAtomPosition,rtol,atol),0)
#        if AtomIndex(FracXYZ,FinalAtomPosition,rtol,atol) > AtomIndex(FracXYZ,InitialAtomPosition,rtol,atol):
#            print ("Final Atom Position > Initial Atom Position")
#            InitialASEXYZ = np.delete(InitialASEXYZ,AtomIndex(FracXYZ,FinalAtomPosition,rtol,atol)-1,0)
#            FinalASEXYZ = np.delete(FinalASEXYZ,AtomIndex(FracXYZ,InitialAtomPosition,rtol,atol),0)
#        if AtomIndex(FracXYZ,FinalAtomPosition,rtol,atol) < AtomIndex(FracXYZ,InitialAtomPosition,rtol,atol):
#            print ("Initial Atom Position > Final Atom Position")
#            InitialASEXYZ = np.delete(InitialASEXYZ,AtomIndex(FracXYZ,FinalAtomPosition,rtol,atol),0)
#            FinalASEXYZ = np.delete(FinalASEXYZ,AtomIndex(FracXYZ,InitialAtomPosition,rtol,atol)-1,0)
#
#    print("Puting Back the Trace Atoms")
#    if frac_or_cart  == True:
#        InitialASEXYZ = np.vstack([InitialASEXYZ,XYZFull[AtomIndex(XYZFull,InitialAtomPosition,rtol,atol)]])
#        FinalASEXYZ = np.vstack([FinalASEXYZ,XYZFull[AtomIndex(XYZFull,FinalAtomPosition,rtol,atol)]])
#    else:
#        InitialASEXYZ = np.vstack([InitialASEXYZ,XYZFull[AtomIndex(FracXYZ,InitialAtomPosition,rtol,atol)]])
#        FinalASEXYZ = np.vstack([FinalASEXYZ,XYZFull[AtomIndex(FracXYZ,FinalAtomPosition,rtol,atol)]])
#
#
#
#    info_ASE = {'initial':InitialASEXYZ, 'final':FinalASEXYZ }
#
#    return info_ASE

#def is_frac_or_cart (Position):
#    if Position.max()>1.0:
#        print("The Atom Positions are in Cartesian (Ang)  " )
#        Frac = False
#    else:
#        print("The Atom Positions are in Internal (fractional)  " )
#        Frac = True
#    return 

