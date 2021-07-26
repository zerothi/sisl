# -*- coding: utf-8 -*-
########################################################################################
# Copyright (c), The SiestaBarriers authors. All rights reserved.                       #
#                                                                                      #
# SiestaBarriers is hosted on GitHub at https://github.com/.................. #
# For further information on the license, see the LICENSE.txt file                     #
########################################################################################
from __future__ import absolute_import

from SiestaBarriers.SiestaBarriersBase import SiestaBarriersBase
from SiestaBarriers.Utils.utils_siesta import read_siesta_fdf,read_siesta_XV
from SiestaBarriers.Utils.utils_ring import pre_prepare_sisl,is_frac_or_cart_or_index,pre_prepare_ase_after_relax
from SiestaBarriers.SiestaBarriersIO import SiestaBarriersIO
import numpy as np 

__author__ = "Arsalan Akhatar"
__copyright__ = "Copyright 2021, SIESTA Group"
__version__ = "0.1"
__maintainer__ = "Arsalan Akhtar"
__email__ = "arsalan_akhtar@outlook.com," + \
        " miguel.pruneda@icn2.cat "
__status__ = "Development"
__date__ = "Janurary 30, 2021"


class Ring(SiestaBarriersBase):
    """
    """
    def __init__(self,
                 #number_of_images = None,
                 host_path = None,
                 host_fdf_name = None,
                 initial_relaxed_path = None,
                 initial_relaxed_fdf_name = None,
                 final_relaxed_path = None,
                 final_relaxed_fdf_name = None,
                 host_structure = None ,
                 initial_structure = None,
                 final_structure = None,
                 #image_direction = None ,
                 ring_atoms_paths = None,
                 ring_atoms_index = None,
                 tolerance_radius = None,
                 flos_path = None,
                 #relaxed = None,
                 #ghost = None ,
                 #interpolation_method = None,
                 atol = 1e-2,
                 rtol = 1e-2
                 ):

        super().__init__(
                host_path,
                )

        self.atol = atol 
        self.rtol = rtol
        self.host_fdf_name = host_fdf_name 
        self.initial_relaxed_fdf_name = initial_relaxed_fdf_name 
        self.final_relaxed_fdf_name = final_relaxed_fdf_name 
        self.tolerance_radius = tolerance_radius

        #NEBBase.__init__(self,
        #                 #number_of_images ,
        #                 host_path,
        #                 initial_relaxed_path,
        #                 final_relaxed_path ,
        #                 host_structure  ,
        #                 initial_structure ,
        #                 final_structure ,
        #                 #image_direction = 'z',
        #                 trace_atom_initial_position ,
        #                 trace_atom_final_position ,
        #                 #ghost ,
        #                 #relaxed ,
        #                 #interpolation_method,
        #                 #image_direction,
        #                 )
    #---------------------------------------------------------
    # Set Methods
    #---------------------------------------------------------
    def set_atol(self,atol):
        """
        """
        self.atol = atol 
    def set_rtol(self,rtol):
        """
        """
        self.rtol = rtol
    def set_host_fdf_name(self,host_fdf_name):
        """
        """
        self.host_fdf_name = host_fdf_name

    def set_initial_relaxed_fdf_name(self,initial_relaxed_fdf_name):
        """
        """
        self.initial_relaxed_fdf_name = initial_relaxed_fdf_name

    def set_final_relaxed_fdf_name(self,final_relaxed_fdf_name):
        """
        """
        self.final_relaxed_fdf_name = final_relaxed_fdf_name
    def set_tolerance_radius(self,tolerance_radius):
        """
        """
        self.tolerance_radius = tolerance_radius
    def set_ring_atoms_index(self,ring_atoms_index):
        """
        """
        self.ring_atoms_index = ring_atoms_index

    def set_ring_atoms_paths(self,ring_atoms_paths):
        """
        """
        self.ring_atoms_paths = ring_atoms_paths
        
    #---------------------------------------------------------
    # Main Methods
    #---------------------------------------------------------
 
    def generate_ring_images(self):
        """

        """
        import sisl
        from ase.neb import NEB 
        if self.relaxed == True:
             print ("=================================================")
             print ("     The Relaxed Ring Image Generation ...   ")
             print ("=================================================")

             self.host_structure = read_siesta_fdf(self.host_path,self.host_fdf_name)
             self.initial_structure = read_siesta_XV(self.initial_relaxed_path,self.initial_relaxed_fdf_name)
             self.final_structure = read_siesta_XV(self.final_relaxed_path,self.final_relaxed_fdf_name)
             
             self.test = pre_prepare_ase_after_relax(self.initial_structure['XV'],self.final_structure['XV'])
                  
             initial = sisl.Geometry.toASE(self.test['initial'])
             final = sisl.Geometry.toASE(self.test['final'])


        else:
             print ("=================================================")
             print ("     The Initial Ring Image Generation ...   ")
             print (" NOTE: ONLY INDEXING RING PATH ALLOWED        ")
             print ("=================================================")
             
             self.host_structure = read_siesta_fdf(self.host_path,self.host_fdf_name)
             
             frac_or_cart_or_index =  'index'   #is_frac_or_cart_or_index(self.ring_atoms_paths[0] )
             self.Test = pre_prepare_sisl(frac_or_cart_or_index,
                                     self.host_structure['Geometry'],
                                     self.ring_atoms_index,
                                     self.ring_atoms_paths , 
                                     self.rtol,
                                     self.atol,
                                     )
             
             self.initial_Dic_sisl={}
             self.final_Dic_sisl={}
             self.initial_Dic={}
             self.final_Dic={}
             for k in range(len(self.ring_atoms_paths)):
                print ("initial_sisl",k)
                self.initial_Dic_sisl[k]=self.Test['initial_sisl'].add(self.Test['initial_dic_sisl'][k])
                self.initial_Dic[k] = sisl.Geometry.toASE(self.initial_Dic_sisl[k])
                print ("final_sisl",k)
                self.final_Dic_sisl[k]=self.Test['initial_sisl'].add(self.Test['final_dic_sisl'][k])
                self.final_Dic[k] = sisl.Geometry.toASE(self.final_Dic_sisl[k])

             self.structure_without_ring_sisl = self.Test['structure_without_ring']
             self.structure_without_ring_ASE = sisl.Geometry.toASE(self.structure_without_ring_sisl)

             
             #--------------------------------
             #
             #---------------------------------

             from ase.neb import NEB
             print("NEB Interpolation")
             images_Dic={}
             neb_Dic={}
             for i in range(len(self.ring_atoms_paths)):
                print ("NEB For Atoms in Ring Movement",i+1)
                images_Dic[i]=[self.initial_Dic[i]]
             #images=[initial]
                for j in range(self.number_of_images):
                    print ("Copying ASE For NEB Image ",j+1)
                    images_Dic[i].append(self.initial_Dic[i].copy())
                images_Dic[i].append(self.final_Dic[i])
             #    #%%
                neb_Dic[i]=NEB(images_Dic[i])
                neb_Dic[i].interpolate('idpp')
             
             self.sisl_images=[]
             NEW = self.structure_without_ring_sisl

             for i in range(self.number_of_images+2):
                NEW = self.structure_without_ring_sisl
                for k in range(len(self.ring_atoms_paths)):
                    print (k)
                    NEW = NEW.add(sisl.Geometry(neb_Dic[k].images[i].get_positions()[-1],
                        atoms=neb_Dic[k].images[i].get_atomic_numbers()[-1]
                       ))
                self.sisl_images.append(NEW)

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
                                          self.final_relaxed_fdf_name,)

             #initial = sisl.Geometry.toASE(self.test['initial'])
             #final = sisl.Geometry.toASE(self.test['final'])

                #self.initial_Dic = sisl.Geometry.toASE(self.initial_Dic_sisl[k])
                #self.final_Dic = sisl.Geometry.toASE(self.final_Dic_sisl[k])


        #%%*****************************************************************
        #%% Do the Image Creation with ASE Here
        #%%*****************************************************************
        #if self.relaxed == True:
        #    print("NEB Interpolation for : Relaxed Structures")
        #else:
        #    print("NEB Interpolation for : UnRelaxed Structures")
        #
        #print ("Copying ASE For NEB Image 0 : initial image ")
        #self.images = [initial]
        #for i in range(self.number_of_images):
        #    print ("Copying ASE For NEB Image {} : images ".format(i+1))
        #    self.images.append(initial.copy())
        #self.images.append(final)
        #print ("Copying ASE For NEB Image {} : final image".format(i+2))
        #%% 
        #self.neb = NEB(self.images)
        #self.neb.interpolate(self.interpolation_method)
 
        
        #self.sisl_images = []
        #for i in range(self.number_of_images+2):
        #    self.sisl_images.append(sisl.Geometry.fromASE(self.images[i]))
        #-------------------------------------------------------------------
        # For Exchange
        #-------------------------------------------------------------------
        #if self.relaxed == True:
        #    #d = self.test['trace_atom_A_final'] - self.test[]
        #    d = self.trace_atom_final_position - self.trace_atom_initial_position
        #else:
        #    d = self.trace_atom_final_position - self.trace_atom_initial_position
        #Steps = d / (self.number_of_images +1)
        # 
        #if self.relaxed == True:
        #    FinalAtomPositionKick = self.test['trace_atom_B_initial'].xyz[0]
        #else:
        #    FinalAtomPositionKick = self.trace_atom_final_position
        #MovingAtomIndex=len(self.neb.images[0].get_positions())
        #MovingAtomKick=np.array([])
        #for l in range(self.neb.nimages):
        #    if l==0:
        #        MovingAtomKick=np.append(MovingAtomKick,FinalAtomPositionKick)
        #    if l>0:
        #        MovingAtomKick=np.append(MovingAtomKick,FinalAtomPositionKick+Steps)
        #        FinalAtomPositionKick=FinalAtomPositionKick+Steps
        #MovingAtomKick=MovingAtomKick.reshape(self.number_of_images+2,3)
        #
        #if self.relaxed == True:
        #    steps_x = np.divide(self.test['trace_atom_B_kicked'].xyz[0][0]-MovingAtomKick[0][0],  len(MovingAtomKick))
        #    steps_y = np.divide(self.test['trace_atom_B_kicked'].xyz[0][1]-MovingAtomKick[0][1],  len(MovingAtomKick))
        #    steps_z = np.divide(self.test['trace_atom_B_kicked'].xyz[0][2]-MovingAtomKick[0][2],  len(MovingAtomKick))
        # 
        #else :
        #    steps_x = np.divide(self.kicked_atom_final_position[0]-MovingAtomKick[0][0],  len(MovingAtomKick))
        #    steps_y = np.divide(self.kicked_atom_final_position[1]-MovingAtomKick[0][1],  len(MovingAtomKick))
        #    steps_z = np.divide(self.kicked_atom_final_position[2]-MovingAtomKick[0][2],  len(MovingAtomKick))
        #print (steps_x)
        #print (steps_y)
        #print (steps_z)
        #Offset
        #Offset = np.array([])
        #for l in range(len(MovingAtomKick)):
        #    if l == 0:
        #        Offset=np.append(Offset,0.0)
        #        Offset=np.append(Offset,0.0)
        #        Offset=np.append(Offset,0.0)
        #    else:
        #        Offset=np.append(Offset,steps_x*l + steps_x)
        #        Offset=np.append(Offset,steps_y*l + steps_y)
        #        Offset=np.append(Offset,steps_z*l + steps_z)
        #Offset=Offset.reshape(len(MovingAtomKick),3)
        #
        #MovingAtomKick=Offset+MovingAtomKick[0]
        #
        #sisl_moving=[]
        #for i in range(self.number_of_images+2):
        #    sisl_moving.append(sisl.Geometry(MovingAtomKick[i],atoms=self.test['trace_atom_B_kicked'].atoms.Z))
        #print(" Putting Kicked Specie in Sisl Geometry Object ")
        # 
        #for i in range(self.number_of_images+2):
        #    self.sisl_images[i] = self.sisl_images[i].add(sisl_moving[i])
        #    self.sisl_images[i] = self.sisl_images[i].add(moving_specie_B[i])


    def write_all_images_sisl(self, fname = 'images' , out_format = 'xyz'):
        """
        """
        for i in range(self.number_of_images+2):
            self.sisl_images[i].write(fname +'-'+str(i)+"."+out_format)

    def write_image_n_sisl(self,n,fname = 'images' , out_format = 'xyz' ):
        """

        """
        self.sisl_images[n].write(fname +'-'+str(n)+"."+out_format)
        print ("DONE!")

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
        
