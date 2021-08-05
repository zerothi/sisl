# -*- coding: utf-8 -*-
########################################################################################
# Copyright (c), The SislSiestaBarriers authors. All rights reserved.                  #
# SislSiestaBarriers is hosted on GitHub at :                                          #
# https://github.com/zerothi/sisl/toolbox/siesta/barriers                              #
# For further information on the license, see the LICENSE.txt file                     #
########################################################################################

from .BarriersBase import SiestaBarriersBase

class Kick(SiestaBarriersBase):
    """
    """
    def __init__(self,
                 host_structure  ,
                 number_of_images ,
                 trace_atom_initial_position ,
                 trace_atom_final_position ,
                 kicked_atom_final_position,
                 ghost = False ,
                 interpolation_method = 'idpp',
                ):


        super().__init__(
                neb_scheme = 'kick',
                relaxed = False,
                host_structure = host_structure ,
                number_of_images = number_of_images,
                initial_relaxed_path = None,
                initial_relaxed_fdf_name = None,
                final_relaxed_path = None,
                final_relaxed_fdf_name = None,
                trace_atom_initial_position  = trace_atom_initial_position,
                trace_atom_final_position = trace_atom_final_position,
                kicked_atom_final_position = kicked_atom_final_position,
                atol = 1e-2,
                rtol = 1e-2
                )
        

        self.host_structure = host_structure
        self.number_of_images = number_of_images
        self.interpolation_method = interpolation_method
        self.ghost = ghost

        
    #---------------------------------------------------------
    # Main Methods
    #---------------------------------------------------------
 
    def Generate_Kick_Images(self):
        """

        """
        from .Utils.utils_siesta import read_siesta_fdf,read_siesta_XV,FixingSislImages
        from .Utils.utils_kick import pre_prepare_sisl,is_frac_or_cart_or_index,pre_prepare_ase_after_relax
        import os
        import glob,shutil
        import numpy as np 
        import sys
        from .BarriersIO import SiestaBarriersIO


        import sisl
        from ase.neb import NEB 
        if self.relaxed == True:
             print ("=================================================")
             print ("     The Relaxed Kick Image Generation ...   ")
             print ("=================================================")
             if self.initial_relaxed_path == None or self.final_relaxed_path == None :
                 sys.exit("intial/final relaxed path not provided")
             if self.initial_relaxed_fdf_name == None or self.final_relaxed_fdf_name == None :
                 sys.exit("intial/final relaxed fdf not provided")

             #self.host_structure = read_siesta_fdf(self.host_path,self.host_fdf_name)
             self.initial_structure = read_siesta_XV(self.initial_relaxed_path,self.initial_relaxed_fdf_name)
             self.final_structure = read_siesta_XV(self.final_relaxed_path,self.final_relaxed_fdf_name)
             
             self.test = pre_prepare_ase_after_relax(self.initial_structure['XV'],self.final_structure['XV'])
                  
             initial = sisl.Geometry.toASE(self.test['initial'])
             final = sisl.Geometry.toASE(self.test['final'])


        else:
             print ("=================================================")
             print ("     The Initial Kick Image Generation ...   ")
             print ("=================================================")
             #self.host_structure = read_siesta_fdf(self.host_path,self.host_fdf_name)
             frac_or_cart_or_index = is_frac_or_cart_or_index(self.trace_atom_initial_position )
             self.test = pre_prepare_sisl(frac_or_cart_or_index,
                                     #self.host_structure['Geometry'],
                                     self.host_structure,
                                     self.trace_atom_initial_position , 
                                     self.trace_atom_final_position,
                                     self.kicked_atom_final_position,
                                     self.rtol,
                                     self.atol,
                                     )
             initial = sisl.Geometry.toASE(self.test['initial'])
             final = sisl.Geometry.toASE(self.test['final'])


        #%%*****************************************************************
        #%% Do the Image Creation with ASE Here
        #%%*****************************************************************
        if self.relaxed == True:
            print("NEB Interpolation for : Relaxed Structures")
        else:
            print("NEB Interpolation for : UnRelaxed Structures")
        
        print ("Copying ASE For NEB Image 0 : initial image ")
        self.images = [initial]
        for i in range(self.number_of_images):
            print ("Copying ASE For NEB Image {} : images ".format(i+1))
            self.images.append(initial.copy())
        self.images.append(final)
        print ("Copying ASE For NEB Image {} : final image".format(i+2))
        #%% 
        self.neb = NEB(self.images)
        self.neb.interpolate(self.interpolation_method)
 
        
        self.sisl_images = []
        for i in range(self.number_of_images+2):
            #self.sisl_images.append(sisl.Geometry.fromASE(self.images[i]))
            temp = sisl.Geometry.fromASE(self.images[i])
            self.sisl_images.append(FixingSislImages(self.test['initial'], temp,"ghost",self.relaxed))
        #-------------------------------------------------------------------
        # For Kick
        #-------------------------------------------------------------------
        if self.relaxed == True:
            #d = self.test['trace_atom_A_final'] - self.test[]
            d = self.trace_atom_final_position - self.trace_atom_initial_position
        else:
            d = self.trace_atom_final_position - self.trace_atom_initial_position
        Steps = d / (self.number_of_images +1)
        
        if self.relaxed == True:
            FinalAtomPositionKick = self.test['trace_atom_B_initial'].xyz[0]
        else:
            FinalAtomPositionKick = self.trace_atom_final_position
        MovingAtomIndex=len(self.neb.images[0].get_positions())
        MovingAtomKick=np.array([])
        for l in range(self.neb.nimages):
            if l==0:
                MovingAtomKick=np.append(MovingAtomKick,FinalAtomPositionKick)
            if l>0:
                MovingAtomKick=np.append(MovingAtomKick,FinalAtomPositionKick+Steps)
                FinalAtomPositionKick=FinalAtomPositionKick+Steps
        MovingAtomKick=MovingAtomKick.reshape(self.number_of_images+2,3)
        
        if self.relaxed == True:
            steps_x = np.divide(self.test['trace_atom_B_kicked'].xyz[0][0]-MovingAtomKick[0][0],  len(MovingAtomKick))
            steps_y = np.divide(self.test['trace_atom_B_kicked'].xyz[0][1]-MovingAtomKick[0][1],  len(MovingAtomKick))
            steps_z = np.divide(self.test['trace_atom_B_kicked'].xyz[0][2]-MovingAtomKick[0][2],  len(MovingAtomKick))
 
        else :
            steps_x = np.divide(self.kicked_atom_final_position[0]-MovingAtomKick[0][0],  len(MovingAtomKick))
            steps_y = np.divide(self.kicked_atom_final_position[1]-MovingAtomKick[0][1],  len(MovingAtomKick))
            steps_z = np.divide(self.kicked_atom_final_position[2]-MovingAtomKick[0][2],  len(MovingAtomKick))
        print (steps_x)
        print (steps_y)
        print (steps_z)
        #Offset
        Offset = np.array([])
        for l in range(len(MovingAtomKick)):
            if l == 0:
                Offset=np.append(Offset,0.0)
                Offset=np.append(Offset,0.0)
                Offset=np.append(Offset,0.0)
            else:
                Offset=np.append(Offset,steps_x*l + steps_x)
                Offset=np.append(Offset,steps_y*l + steps_y)
                Offset=np.append(Offset,steps_z*l + steps_z)
        Offset=Offset.reshape(len(MovingAtomKick),3)
        
        MovingAtomKick=Offset+MovingAtomKick[0]
        self.MovingAtomKick = MovingAtomKick
        print("DEBUG: {}".format(self.MovingAtomKick))
        sisl_moving=[]
       


       # Fixing the Tag
        self.KickedAtomInfo = self.test['trace_atom_B_kicked']
        print("DEBUG: {}".format(self.KickedAtomInfo))
        #KickedAtomInfo = sisl.Atom(self.test['trace_atom_B_kicked'].atoms.Z)

        for i in range(self.number_of_images+2):
            #sisl_moving.append(sisl.Geometry(MovingAtomKick[i],atoms= sisl.Atom (KickedAtomInfo.Z,tag=KickedAtomInfo.symbol+"_kicked")))
            sisl_moving.append(sisl.Geometry(xyz = self.MovingAtomKick[i],
                                             atoms = sisl.Atom(Z = self.KickedAtomInfo.atom[0].Z,tag=self.KickedAtomInfo.atoms.atom[0].symbol+"_kicked")))
            #sisl_moving.append(sisl.Geometry(MovingAtomKick[i],atoms=self.test['trace_atom_B_kicked'].atoms.Z))
        print(" Putting Kicked Specie in Sisl Geometry Object ")
        
        for i in range(self.number_of_images+2):
            self.sisl_images[i] = self.sisl_images[i].add(sisl_moving[i])
        #    self.sisl_images[i] = self.sisl_images[i].add(moving_specie_B[i])

        self.IO = SiestaBarriersIO(neb_type = 'kick',
                                   sisl_images = self.sisl_images,
                                   flos_path =  self.flos_path,
                                   flos_file_name_relax = self.flos_file_name_relax,
                                   flos_file_name_neb =  self.flos_file_name_neb,
                                   number_of_images = self.number_of_images,
                                   initial_relaxed_path = self.initial_relaxed_path,
                                   final_relaxed_path = self.final_relaxed_path,
                                   initial_relaxed_fdf_name = self.final_relaxed_path,
                                   final_relaxed_fdf_name = self.final_relaxed_fdf_name,
                                   relax_engine = self.relax_engine,
                                   relaxed = self.relaxed,
                                   ghost = self.ghost,
                                          )



    def NEB_Results(self):
        """
        """
        self.IO = SiestaBarriersIO(self.sisl_images,
                                          self.flos_path,
                                          self.flos_file_name_relax,
                                          self.flos_file_name_neb,
                                          self.number_of_images,
                                          self.initial_relaxed_path,
                                          self.final_relaxed_path,
                                          self.relax_engine,
                                          self.relaxed,
                                          self.ghost,
                                          self.initial_relaxed_fdf_name,
                                          self.final_relaxed_fdf_name,
                                          self.neb_results_path
                                          )

    #=========================================================================
    #  Checking Methods
    #=========================================================================

    def check_ghost(self):
        """
        """
        for i in self.initial_structure['XV'].atoms.Z:
            if i < 0 :
                self.ghost = True
                print("There are ghost Species in 'XV'!")

    def check_AA_or_AB(self):
        """
        """
        if self.initial_structure['XV'].atoms.Z[-1] == self.initial_structure['XV'].atoms.Z[-2]:
            print ("The Exchange Species Are Same Atoms!")
            return True
        else:
            print ("The Exchange Species Are Different Atoms!")
            return False
        
