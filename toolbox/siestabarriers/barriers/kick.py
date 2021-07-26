# -*- coding: utf-8 -*-
########################################################################################
# Copyright (c), The SiestaBarriers authors. All rights reserved.                       #
#                                                                                      #
# SiestaBarriers is hosted on GitHub at https://github.com/.................. #
# For further information on the license, see the LICENSE.txt file                     #
########################################################################################
from __future__ import absolute_import

from SiestaBarriers.SiestaBarriersBase import SiestaBarriersBase
from SiestaBarriers.Utils.utils_siesta import read_siesta_fdf,read_siesta_XV,FixingSislImages
from SiestaBarriers.Utils.utils_kick import pre_prepare_sisl,is_frac_or_cart_or_index,pre_prepare_ase_after_relax
import os
import glob,shutil
import numpy as np 
import sys
from SiestaBarriers.SiestaBarriersIO import SiestaBarriersIO

__author__ = "Arsalan Akhatar"
__copyright__ = "Copyright 2021, SIESTA Group"
__version__ = "0.1"
__maintainer__ = "Arsalan Akhtar"
__email__ = "arsalan_akhtar@outlook.com," + \
        " miguel.pruneda@icn2.cat "
__status__ = "Development"
__date__ = "Janurary 30, 2021"


class Kick(SiestaBarriersBase):
    """
    """
    def __init__(self,
                 host_path = None,
                 host_fdf_name = None,
                 host_structure = None ,
                 initial_relaxed_path = None,
                 initial_relaxed_fdf_name = None,
                 initial_structure = None,
                 final_relaxed_path = None,
                 final_relaxed_fdf_name = None,
                 final_structure = None,
                 trace_atom_initial_position = None,
                 trace_atom_final_position = None,
                 tolerance_radius = None,
                 kicked_atom_final_position = None,
                 flos_path = None,
                 flos_file_name_relax = None,
                 flos_file_name_neb = None,
                 #relaxed = None,
                 #ghost = None ,
                 #interpolation_method = None,
                 atol = 1e-2,
                 rtol = 1e-2
                 ):


        super().__init__(
                host_path,
                host_fdf_name,
                host_structure,
                initial_relaxed_path,
                initial_relaxed_fdf_name,
                initial_structure,
                final_relaxed_path,
                final_relaxed_fdf_name,
                final_structure,
                trace_atom_initial_position,
                trace_atom_final_position,
                )
        

        self.atol = atol 
        self.rtol = rtol
        #self.host_path = host_path
        #self.host_fdf_name = host_fdf_name 
        #self.host_structure = host_structure
        self.initial_relaxed_path = initial_relaxed_path
        self.initial_relaxed_fdf_name = initial_relaxed_fdf_name 
        #self.initial_structure = initial_structure        
        self.final_relaxed_path = final_relaxed_path 
        self.final_relaxed_fdf_name = final_relaxed_fdf_name 
        #self.final_structure = final_structure
        self.tolerance_radius = tolerance_radius
        #self.kicked_atom_final_position = kicked_atom_final_position
        
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
     
    def set_initial_relaxed_path(self,initial_relaxed_path):
        """
        """
        self.initial_relaxed_path = initial_relaxed_path
    def set_final_relaxed_path(self,final_relaxed_path):
        """
        """
        self.final_relaxed_path = final_relaxed_path


    #---------------------------------------------------------
    # Main Methods
    #---------------------------------------------------------
 
    def generate_kick_images(self):
        """

        """
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
             self.host_structure = read_siesta_fdf(self.host_path,self.host_fdf_name)
             frac_or_cart_or_index = is_frac_or_cart_or_index(self.trace_atom_initial_position )
             self.test = pre_prepare_sisl(frac_or_cart_or_index,
                                     self.host_structure['Geometry'],
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

        sisl_moving=[]
        
        # Fixing the Tag
        KickedAtomInfo = sisl.Atom(self.test['trace_atom_B_kicked'].atoms.Z)

        for i in range(self.number_of_images+2):
            sisl_moving.append(sisl.Geometry(MovingAtomKick[i],atoms= sisl.Atom (KickedAtomInfo.Z,tag=KickedAtomInfo.symbol+"_kicked")))
            #sisl_moving.append(sisl.Geometry(MovingAtomKick[i],atoms=self.test['trace_atom_B_kicked'].atoms.Z))
        print(" Putting Kicked Specie in Sisl Geometry Object ")
        
        for i in range(self.number_of_images+2):
            self.sisl_images[i] = self.sisl_images[i].add(sisl_moving[i])
        #    self.sisl_images[i] = self.sisl_images[i].add(moving_specie_B[i])

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
    #  Writing Methods
    #=========================================================================

    #def write_all_images_sisl(self, fname = 'images' , out_format = 'xyz'):
    #    """
    #    """
    #    for i in range(self.number_of_images+2):
    #        self.sisl_images[i].write(fname +'-'+str(i)+"."+out_format)

    #def write_image_n_sisl(self,n,fname = 'images' , out_format = 'xyz' ):
    #    """
    #
    #    """
    #    self.sisl_images[n].write(fname +'-'+str(n)+"."+out_format)
    #    print ("DONE!")

    #def prepare_endpoint_relax(self, folder_name="image", fname = 'input' , out_format = 'fdf'):
    #
    #    """
    #    """
    #    if self.relaxed == True :
    #        print (" The Relaxed Flag is True endpoint relaxation PASS...!")
    #        pass
    #    else:
    #        final_image_n = self.number_of_images +  1
    #        if os.path.isdir(folder_name+"-0"):
    #            print (" The Image 0  Folder is there Already PASS")
    #            print (" Check The Folder: '{}' ".format(folder_name+"-0"))
    #            pass
    #        else:
    #            os.mkdir(folder_name+"-0")
    #            os.chdir(folder_name+"-0")
    #            self.sisl_images[0].write(fname+'.fdf')
    #            if self.relax_engine == 'lua':
    #                shutil.copy(self.flos_path + self.flos_file_name_relax,'./')
    #            os.chdir('../')
    #        if os.path.isdir(folder_name+"-"+str(final_image_n)):
    #            print (" The Image {}  Folder is there".format(final_image_n))
    #            print (" Check The Folder: '{}' ".format(folder_name+"-"+str(final_image_n)))
    #            pass
    #        else:
    #            os.mkdir(folder_name+"-"+str(final_image_n))
    #            os.chdir(folder_name+"-"+str(final_image_n))
    #            self.sisl_images[final_image_n].write(fname+'.fdf')
    #            if self.relax_engine == 'lua':
    #                shutil.copy(self.flos_path + self.flos_file_name_relax,'./')
    #            os.chdir('../')

    #def prepare_neb(self,folder_name='neb'):
    #    """
    #    """
    #    #import glob,shutil
    #    if os.path.isdir(folder_name):
    #        print (" The NEB Folder is there Already PASS")
    #        print (" Check The Folder: '{}' ".format(folder_name))
    #        pass
    #    else:
    #        os.mkdir(folder_name)
    #        os.chdir(folder_name)
    #        self.sisl_images[0].write('input.fdf')
    #        self.write_all_images_sisl()
    #
    #        for file in glob.glob(self.initial_relaxed_path+"/*.DM"):
    #            print("Copying DM 0  ...")
    #            print(file)
    #            shutil.copy(file,'./NEB.DM.0')
    #        for file in glob.glob(self.final_relaxed_path+"/*.DM"):
    #            print("Copying DM {} ... ".format(self.number_of_images+1))
    #            print(file)
    #            shutil.copy(file,'./NEB.DM.{}'.format(self.number_of_images+1))
    #
    #        shutil.copy(self.flos_path + self.flos_file_name_neb,'./')
    #        os.chdir('../')
    #        print("NEB Folder Ready to Run!")
  


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
        
