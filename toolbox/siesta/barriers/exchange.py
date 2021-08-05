# -*- coding: utf-8 -*-
########################################################################################
# Copyright (c), The SislSiestaBarriers authors. All rights reserved.                  #
# SislSiestaBarriers is hosted on GitHub at :                                          #
# https://github.com/zerothi/sisl/toolbox/siesta/barriers                              #
# For further information on the license, see the LICENSE.txt file                     #
########################################################################################

from .BarriersBase import SiestaBarriersBase

class Exchange(SiestaBarriersBase):
    """
    """
    def __init__(self,
                 host_structure ,
                 number_of_images ,
                 trace_atom_initial_position ,
                 trace_atom_final_position ,
                 exchange_direction = 'xyz',
                 tolerance_radius = [0.5,0.5,0.5],
                 ghost = False ,
                 interpolation_method = 'idpp',
                 ):
        
        super().__init__( 
                neb_scheme = 'exchange',
                relaxed = False,
                host_structure = host_structure,
                initial_relaxed_path = None,
                initial_relaxed_fdf_name = None,
                final_relaxed_path = None,
                final_relaxed_fdf_name = None ,
                trace_atom_initial_position = trace_atom_initial_position ,
                trace_atom_final_position = trace_atom_final_position ,
                atol = 1e-2,
                rtol = 1e-2
                )
        self.host_structure = host_structure
        self.number_of_images = number_of_images
        self.tolerance_radius = tolerance_radius
        self.trace_atom_initial_position  = trace_atom_initial_position
        self.trace_atom_final_position = trace_atom_final_position
        self.exchange_direction = exchange_direction 
        self.tolerance_radius = tolerance_radius
        self.ghost = ghost
        self.interpolation_method = interpolation_method

    #---------------------------------------------------------
    # Main Methods
    #---------------------------------------------------------
 
    def Generate_Exchange_Images(self):
        """

        """
        from .Utils.utils_siesta import read_siesta_fdf,read_siesta_XV
        from .Utils.utils_exchange import pre_prepare_sisl,is_frac_or_cart_or_index,pre_prepare_ase_after_relax_AA,pre_prepare_ase_after_relax_AB,pre_prepare_ase_after_relax_A,pre_prepare_ase_after_relax_B

        import os
        import glob,shutil
        import numpy as np
        import sys
        from .BarriersIO import SiestaBarriersIO
        import sisl
        from ase.neb import NEB 

        if self.relaxed == True:
             print ("=================================================")
             print ("     The Relaxed Exchange Image Generation ...   ")
             print ("=================================================")

             #self.host_structure = read_siesta_fdf(self.host_path,self.host_fdf_name)
             self.initial_structure = read_siesta_XV(self.initial_relaxed_path,self.initial_relaxed_fdf_name)
             self.final_structure = read_siesta_XV(self.final_relaxed_path,self.final_relaxed_fdf_name)
             
             if self.check_AA_or_AB() == True:
                 self.test_A = pre_prepare_ase_after_relax_A(self.initial_structure['XV'],self.final_structure['XV'])
                 self.test_B = pre_prepare_ase_after_relax_B(self.initial_structure['XV'],self.final_structure['XV'])
             else:
                 #print ("Not Implemented Yet!")
                 self.test_A = pre_prepare_ase_after_relax_A(self.initial_structure['XV'],self.final_structure['XV'])
                 self.test_B = pre_prepare_ase_after_relax_B(self.initial_structure['XV'],self.final_structure['XV'])
             initial_A = sisl.Geometry.toASE(self.test_A['initial'])
             final_A = sisl.Geometry.toASE(self.test_A['final'])
             initial_B = sisl.Geometry.toASE(self.test_B['initial'])
             final_B = sisl.Geometry.toASE(self.test_B['final'])


        else:
             print ("=================================================")
             print ("     The Initial Exchange Image Generation ...   ")
             print ("=================================================")

             frac_or_cart_or_index = is_frac_or_cart_or_index(self.trace_atom_initial_position )
             
             self.test = pre_prepare_sisl(frac_or_cart_or_index,
                                     self.host_structure,
                                     self.trace_atom_initial_position , 
                                     self.trace_atom_final_position,
                                     self.rtol,
                                     self.atol,
                                     )
             initial = sisl.Geometry.toASE(self.test['initial'])
             final = sisl.Geometry.toASE(self.test['final'])
             
             self.initial_structure = {}
             self.final_structure = {}
             self.initial_structure ['XV'] = self.test['initial']
             self.final_structure ['XV'] = self.test['final']
             self.check_AA_or_AB()

        #%%*****************************************************************
        #%% Do the Image Creation with ASE Here
        #%%*****************************************************************
        if self.relaxed == True:
            print("NEB Interpolation for : Relaxed Structures")
            print ("Copying ASE For NEB Image 0 : initial image ")
            self.images_A = [initial_A]
            self.images_B = [initial_B]
            for i in range(self.number_of_images):
                print ("Copying ASE For NEB Image {} : images ".format(i+1))
                self.images_A.append(initial_A.copy())
                self.images_B.append(initial_B.copy())
            self.images_A.append(final_A)
            self.images_B.append(final_B)
            print ("Copying ASE For NEB Image {} : final image".format(i+2))
            #%% 
            self.neb_A = NEB(self.images_A)
            self.neb_B = NEB(self.images_B)
            self.neb_A.interpolate(self.interpolation_method)
            self.neb_B.interpolate(self.interpolation_method)
 
        
            self.sisl_images_A = []
            self.sisl_images_B = []
            for i in range(self.number_of_images+2):
                self.sisl_images_A.append(sisl.Geometry.fromASE(self.images_A[i]))
                self.sisl_images_B.append(sisl.Geometry.fromASE(self.images_B[i]))
            #-------------------------------------------------------------------
            # For Exchange
            #-------------------------------------------------------------------
        
            moving_specie_A = {}
            moving_specie_B = {}
            for i in range(len(self.sisl_images_A)):
                moving_specie_A[i]=sisl.Geometry(self.sisl_images_A[i].xyz[-1],atoms=self.sisl_images_A[i].atoms.Z[-1])
                moving_specie_B[i]=sisl.Geometry(self.sisl_images_B[i].xyz[-1],atoms=self.sisl_images_B[i].atoms.Z[-1])
            ni_list=[]
            for i in range(self.number_of_images+2):
                ni_list.append(int(i))
            ni_list.sort(reverse=True)    
            j =0
            #for i in ni_list:
            #    #moving_specie_B[j] = sisl.Geometry(moving_specie_A[i].xyz,atoms=self.test['trace_atom_B_initial'].atoms.Z)
            #    moving_specie_B[j] = sisl.Geometry(moving_specie_A[i].xyz,atoms=self.test_B['trace_atom_B_initial'].atoms.Z)
            #    j = j+1
            #moving_specie_B[0].xyz = self.test_B['trace_atom_B_initial'].xyz
            #moving_specie_B[self.number_of_images+2] =  self.test['trace_atom_B_final'].xyz 
            d = moving_specie_B[0].xyz-moving_specie_A[0].xyz
            d = d.reshape(3,)
            dsq = d**2
            dsum = np.sum(dsq)
            dsum = np.sqrt(dsum)
            ToleranceRadius = self.tolerance_radius
            #ToleranceRadius= np.array([-1.5,1.5,0.0])  #Ang
            print ("d",d)
            print ("dsum",dsum)
            print("Tolernace Radius :{}".format(self.tolerance_radius)) 
            Migration_Direction = self.exchange_direction
            NumberOfImages = self.number_of_images 
        
            if Migration_Direction.lower()=="x":
                d[0]=d[0]+dsum + ToleranceRadius[0]
                Steps=d/NumberOfImages
                print("Migration Direction Axis: x")
            if Migration_Direction.lower()=="y":
                d[1]=d[1]+dsum + ToleranceRadius[1]
                Steps=d/NumberOfImages
                print("Migration Direction Axis: y")
            if Migration_Direction.lower()=="z":
                d[2]=d[2]+dsum + ToleranceRadius[2]
                Steps=d/NumberOfImages
                print("Migration Direction Axis: z")
            if Migration_Direction.lower()=="xy":
                d[0]=d[0]+dsum + ToleranceRadius[0]
                d[1]=d[1]+dsum + ToleranceRadius[1]
                Steps=d/NumberOfImages
                print("Migration Direction Plane: xy")
            if Migration_Direction.lower()=="xz":
                d[0]=d[0]+dsum + ToleranceRadius[0]
                d[2]=d[2]+dsum + ToleranceRadius[2]
                Steps=d/NumberOfImages
                print("Migration Direction Plane: xz")
            if Migration_Direction.lower()=="yz":
                d[1]=d[1]+dsum + ToleranceRadius[1]
                d[2]=d[2]+dsum + ToleranceRadius[2] 
                Steps=d/NumberOfImages
                print("Migration Direction Plane: yz")
            if Migration_Direction.lower()=="xyz":
                d[0]=d[0]+dsum + ToleranceRadius[0]
                d[1]=d[1]+dsum + ToleranceRadius[1]
                d[2]=d[2]+dsum + ToleranceRadius[2]
                Steps=d/NumberOfImages
                print("Migration Direction Volume: xyz")
        
            for l in range(1,NumberOfImages+1):
                print ("Atom A ",l,moving_specie_A[l].xyz+Steps)
                moving_specie_A[l].xyz= moving_specie_A[l].xyz+Steps
            for l in range(1,NumberOfImages+1):
                print ("Atom B ",l,moving_specie_B[l].xyz-Steps)
                moving_specie_B[l].xyz = moving_specie_B[l].xyz-Steps
       
            #print("DEBUG: {}".format(moving_specie_B[self.number_of_images+2])) 
            #moving_specie_B[self.number_of_images+2] = self.test['trace_atom_B_final'].xyz 
            #print("DEBUG: {}".format(moving_specie_B[self.number_of_images+2])) 

            for k in range(self.number_of_images+2):
                self.sisl_images_A[k] = self.sisl_images_A[k].remove([-1])

            print(" Putting Specie A & B in Sisl Geometry Object ")
        
            for i in range(self.number_of_images+2):
                self.sisl_images_A[i] = self.sisl_images_A[i].add(moving_specie_A[i])
                self.sisl_images_A[i] = self.sisl_images_A[i].add(moving_specie_B[i])

            self.sisl_images  = self.sisl_images_A 
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
                self.sisl_images.append(sisl.Geometry.fromASE(self.images[i]))
            #-------------------------------------------------------------------
            # For Exchange
            #-------------------------------------------------------------------
        

            moving_specie_A = {}
            moving_specie_B = {}
            for i in range(len(self.sisl_images)):
                moving_specie_A[i]=sisl.Geometry(self.sisl_images[i].xyz[-1],atoms=self.sisl_images[i].atoms.Z[-1])
            ni_list=[]
            for i in range(self.number_of_images+2):
                ni_list.append(int(i))
            ni_list.sort(reverse=True)
            j =0
            for i in ni_list:
                moving_specie_B[j] = sisl.Geometry(moving_specie_A[i].xyz,atoms=self.test['trace_atom_B_initial'].atoms.Z)
                #moving_specie_B[j] = sisl.Geometry(moving_specie_A[i].xyz,atoms=self.test['trace_atom_B_initial'].atoms.Z)
                j = j+1
            #moving_specie_B[0].xyz = self.test['trace_atom_B_initial'].xyz
            #moving_specie_B[self.number_of_images+2] =  self.test['trace_atom_B_final'].xyz 
            d = moving_specie_B[0].xyz-moving_specie_A[0].xyz
            d = d.reshape(3,)
            dsq = d**2
            dsum = np.sum(dsq)
            dsum = np.sqrt(dsum)
            ToleranceRadius = self.tolerance_radius
            #ToleranceRadius= np.array([-1.5,1.5,0.0])  #Ang
            print ("d",d)
            print ("dsum",dsum)
            print("Tolernace Radius :{}".format(self.tolerance_radius))
            Migration_Direction = self.exchange_direction
            NumberOfImages = self.number_of_images

            if Migration_Direction.lower()=="x":
                d[0]=d[0]+dsum + ToleranceRadius[0]
                Steps=d/NumberOfImages
                print("Migration Direction Axis: x")
            if Migration_Direction.lower()=="y":
                d[1]=d[1]+dsum + ToleranceRadius[1]
                Steps=d/NumberOfImages
                print("Migration Direction Axis: y")
            if Migration_Direction.lower()=="z":
                d[2]=d[2]+dsum + ToleranceRadius[2]
                Steps=d/NumberOfImages
                print("Migration Direction Axis: z")
            if Migration_Direction.lower()=="xy":
                d[0]=d[0]+dsum + ToleranceRadius[0]
                d[1]=d[1]+dsum + ToleranceRadius[1]
                Steps=d/NumberOfImages
                print("Migration Direction Plane: xy")
            if Migration_Direction.lower()=="xz":
                d[0]=d[0]+dsum + ToleranceRadius[0]
                d[2]=d[2]+dsum + ToleranceRadius[2]
                Steps=d/NumberOfImages
                print("Migration Direction Plane: xz")
            if Migration_Direction.lower()=="yz":
                d[1]=d[1]+dsum + ToleranceRadius[1]
                d[2]=d[2]+dsum + ToleranceRadius[2]
                Steps=d/NumberOfImages
                print("Migration Direction Plane: yz")
            if Migration_Direction.lower()=="xyz":
                d[0]=d[0]+dsum + ToleranceRadius[0]
                d[1]=d[1]+dsum + ToleranceRadius[1]
                d[2]=d[2]+dsum + ToleranceRadius[2]
                Steps=d/NumberOfImages
                print("Migration Direction Volume: xyz")
            for l in range(1,NumberOfImages+1):
                print ("Atom A ",l,moving_specie_A[l].xyz+Steps)
                moving_specie_A[l].xyz= moving_specie_A[l].xyz+Steps
            for l in range(1,NumberOfImages+1):
                print ("Atom B ",l,moving_specie_B[l].xyz-Steps)
                moving_specie_B[l].xyz = moving_specie_B[l].xyz-Steps

            #print("DEBUG: {}".format(moving_specie_B[self.number_of_images+2])) 
            #moving_specie_B[self.number_of_images+2] = self.test['trace_atom_B_final'].xyz
            #print("DEBUG: {}".format(moving_specie_B[self.number_of_images+2]))

            for k in range(self.number_of_images+2):
                self.sisl_images[k] = self.sisl_images[k].remove([-1])

            print(" Putting Specie A & B in Sisl Geometry Object ")

            for i in range(self.number_of_images+2):
                self.sisl_images[i] = self.sisl_images[i].add(moving_specie_A[i])
                self.sisl_images[i] = self.sisl_images[i].add(moving_specie_B[i])

        self.IO = SiestaBarriersIO( neb_type = 'exchange',
                                    sisl_images = self.sisl_images,
                                    flos_path = self.flos_path,
                                    flos_file_name_relax = self.flos_file_name_relax,
                                    flos_file_name_neb = self.flos_file_name_neb,
                                    number_of_images = self.number_of_images,
                                    initial_relaxed_path = self.initial_relaxed_path,
                                    initial_relaxed_fdf_name = self.initial_relaxed_fdf_name,
                                    final_relaxed_path = self.final_relaxed_path ,
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
        from .Utils.utils_siestabarrier import is_frac_or_cart_or_index
        if self.relaxed :
            if self.initial_structure['XV'].atoms.Z[-1] == self.initial_structure['XV'].atoms.Z[-2]:
                print ("!----------------------------------!")
                print ("The Exchange Species Are Same Atoms!")
                print ("!----------------------------------!")
                return True
            else:
                print ("!----------------------------------!")
                print ("The Exchange Species Are Different Atoms!")
                print ("!----------------------------------!")
                return False
        else:
            print("DEBUG:")
            if self.test['trace_atom_A_initial'].atoms.Z[0] == self.test['trace_atom_B_initial'].atoms.Z[0]:
                print ("!----------------------------------!")
                print ("The Exchange Species Are Same Atoms!")
                print ("!----------------------------------!")
            else:
                print ("!----------------------------------!")
                print ("The Exchange Species Are Different Atoms!")
                print ("!----------------------------------!")
 
        
