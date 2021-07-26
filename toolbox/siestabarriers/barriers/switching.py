# -*- coding: utf-8 -*-
########################################################################################
# Copyright (c), The SiestaBarriers authors. All rights reserved.                       #
#                                                                                      #
# SiestaBarriers is hosted on GitHub at https://github.com/.................. #
# For further information on the license, see the LICENSE.txt file                     #
########################################################################################
from __future__ import absolute_import

from neb_base import NEBBase
from utils_siesta import read_siesta_fdf,read_siesta_XV
from utils_general import pre_prepare_sisl,is_frac_or_cart_or_index,pre_prepare_ase_after_relax
import os
import glob,shutil
import numpy as np
import sys
 

__author__ = "Arsalan Akhatar"
__copyright__ = "Copyright 2021, SIESTA Group"
__version__ = "0.1"
__maintainer__ = "Arsalan Akhtar"
__email__ = "arsalan_akhtar@outlook.com," + \
        " miguel.pruneda@icn2.cat "
__status__ = "Development"
__date__ = "Janurary 30, 2021"


class Switching(NEBBase):
    """
    """
    def __init__(self,
                 number_of_images = None,
                 #host_path = None,
                 host_fdf_name = None,
                 #initial_relaxed_path = None,
                 initial_relaxed_fdf_name = None,
                 #final_relaxed_path = None,
                 final_relaxed_fdf_name = None,
                 host_structure = None ,
                 initial_structure = None,
                 final_structure = None,
                 image_direction = None ,
                 trace_atom_initial_position = None,
                 trace_atom_final_position = None,
                 relaxed = None,
                 ghost = None ,
                 interpolation_method = None,
                 atol = 1e-2,
                 rtol = 1e-2
                 ):

        self.atol = atol 
        self.rtol = rtol
        self.host_fdf_name = host_fdf_name 
        self.initial_relaxed_fdf_name = initial_relaxed_fdf_name 
        self.final_relaxed_fdf_name = final_relaxed_fdf_name 

        NEBBase.__init__(self,
                         number_of_images ,
                         #host_path,
                         #initial_relaxed_path,
                         #final_relaxed_path ,
                         host_structure  ,
                         initial_structure ,
                         final_structure ,
                         image_direction ,
                         trace_atom_initial_position ,
                         trace_atom_final_position ,
                         ghost ,
                         relaxed ,
                         interpolation_method,
                         )
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

    #---------------------------------------------------------
    # Main Methods
    #---------------------------------------------------------
 
    def generate_switching_images(self):
        """

        """
        import sisl
        from ase.neb import NEB 
        if self.relaxed == True:
             print ("=================================================")
             print ("The Relaxed Vacancy Exchange Image Generation ...")
             print ("=================================================")

             self.host_structure = read_siesta_fdf(self.host_path,self.host_fdf_name)
             self.initial_structure = read_siesta_XV(self.initial_relaxed_path,self.initial_relaxed_fdf_name)
             self.final_structure = read_siesta_XV(self.final_relaxed_path,self.final_relaxed_fdf_name)
             
             self.check_ghost()
             self.test = pre_prepare_ase_after_relax(self.initial_structure['XV'],self.final_structure['XV'],self.ghost)
             initial = sisl.Geometry.toASE(self.test['initial'])
             final = sisl.Geometry.toASE(self.test['final'])


        else:
             print ("=================================================")
             print ("The Initial Vacancy Exchange Image Generation ...")
             print ("=================================================")
             #self.host_structure = read_siesta_fdf(self.host_path,self.host_fdf_name)
             #self.initial_structure = read_siesta_fdf (self.initial_relaxed_path

             initial = sisl.Geometry.toASE(self.initial_structure)
             final = sisl.Geometry.toASE(self.final_structure)


        #%%*****************************************************************
        #%% Do the Image Creation with ASE Here
        #%%*****************************************************************
        if self.relaxed == True:
            print("NEB Interpolation for : Relaxed Structures")
        else:
            print("NEB Interpolation for : UnRelaxed Structures")
        self.images = [initial]
        for i in range(self.number_of_images):
            print ("Copying ASE For NEB Image ",i)
            self.images.append(initial.copy())
        self.images.append(final)
        print ("Copying ASE For NEB Image ",i+1)
        #%% 
        self.neb = NEB(self.images)
        #self.neb.interpolate(self.interpolation_method)
        self.neb.interpolate()
 
        
        self.sisl_images = []
        for i in range(self.number_of_images+2):
            self.sisl_images.append(sisl.Geometry.fromASE(self.images[i]))
        
        if self.ghost == True:
            print(" Putting Ghost in Sisl Geometry Object ")
            for i in range(self.number_of_images+2):
                self.sisl_images[i] = self.sisl_images[i].add(self.test['Ghost_initial'])
                self.sisl_images[i] = self.sisl_images[i].add(self.test['Ghost_final'])


        self.IO = SiestaBarriersIO(self.sisl_images,
                                          self.flos_path,
                                          self.flos_file_name_relax,
                                          self.flos_file_name_neb,
                                          self.number_of_images,
                                          self.initial_relaxed_path,
                                          self.final_relaxed_path,
                                          self.relax_engine,
                                          self.relaxed,
                                          self.ghost
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

