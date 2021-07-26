# -*- coding: utf-8 -*-
########################################################################################
# Copyright (c), The SiestaBarriers authors. All rights reserved.                       #
#                                                                                      #
# SiestaBarriers is hosted on GitHub at https://github.com/.................. #
# For further information on the license, see the LICENSE.txt file                     #
########################################################################################
#from __future__ import absolute_import
from SiestaBarriers.SiestaBarriersBase import SiestaBarriersBase
from SiestaBarriers.SiestaBarriersIO import SiestaBarriersIO
import sisl

__author__ = "Arsalan Akhatar"
__copyright__ = "Copyright 2021, SIESTA Group"
__version__ = "0.1"
__maintainer__ = "Arsalan Akhtar"
__email__ = "arsalan_akhtar@outlook.com," + \
        " miguel.pruneda@icn2.cat "
__status__ = "Development"
__date__ = "Janurary 30, 2021"


class Manual(SiestaBarriersBase):
    """
    """
    def __init__(self,
                 initial_relaxed_fdf_name = None ,
                 initial_relaxed_path = None,
                 initial_structure = None,
                 final_relaxed_fdf_name = None,
                 final_relaxed_path = None,
                 final_structure = None,
                 relaxed = None,
                 ghost = None ,
                 interpolation_method = "idpp",
                 atol = 1e-2,
                 rtol = 1e-2,
                 flos_path = None,
                 ):

        super().__init__(
                 initial_relaxed_fdf_name,
                 initial_relaxed_path ,
                 initial_structure ,
                 final_relaxed_fdf_name ,
                 final_relaxed_path ,
                 final_structure ,
                 relaxed ,
                 ghost ,
                 atol ,
                 rtol ,
                 flos_path,
                )
        
        self.atol = atol 
        self.rtol = rtol
        self.interpolation_method = interpolation_method

    def set_host_path(self,host_path):
        """
        """
        self.host_path = host_path 



    def generate_manual_images(self):
        """

        """
        #from Utils.utils_siesta import read_siesta_fdf
        #from Utils.utils import prepare_ase,is_frac_or_cart,ASEInitilizer
        from ase.neb import NEB 
        print ("The Manual Image Generation ...")
        
        initial = sisl.Geometry.toASE(self.initial_structure)
        final = sisl.Geometry.toASE(self.final_structure)


        #%%*****************************************************************
        #%% Do the Image Creation with ASE Here
        #%%*****************************************************************
        print("NEB Interpolation")

        self.images = [initial]
        for i in range(self.number_of_images):
            print ("Copying ASE For NEB Image ",i)
            self.images.append(initial.copy())
        self.images.append(final)
        print ("Copying ASE For NEB Image ",i+1)
        #%% 
        self.neb = NEB(self.images)
        self.neb.interpolate(self.interpolation_method,mic=True)
        #self.neb.interpolate()

        #images = [initial]
        #for i in range(self.number_of_images):
        #    print ("Copying ASE For NEB Image ",i)
        #    images.append(initial.copy())
        #images.append(final)
        #print ("Copying ASE For NEB Image ",i+1)
        ##%% 
        #self.neb = NEB(images)
        #self.neb.interpolate(self.interpolation_method)
 
        self.sisl_images = []
        for i in range(self.number_of_images+2):
            self.sisl_images.append(sisl.Geometry.fromASE(self.images[i]))
        
        self.IO = SiestaBarriersIO(self.sisl_images,
                                          self.flos_path,
                                          self.flos_file_name_relax,
                                          self.flos_file_name_neb,
                                          self.number_of_images,
                                          self.initial_relaxed_path,
                                          self.initial_relaxed_fdf_name,
                                          self.final_relaxed_path,
                                          self.final_relaxed_fdf_name,
                                          self.relax_engine,
                                          self.relaxed,
                                          self.ghost
                                          )



    def write_all_images(self, fname = 'images' , out_format = 'xyz'):
        """
        """
        for i in range(self.number_of_images+1):
            self.neb.images[i].write(fname +'-'+str(i)+"."+out_format,out_format)

    def write_image_n(self,n,fname = 'images' , out_format ='xyz'):
        """

        """
        self.neb.images[n].write(fname +'-'+str(n)+"."+out_format,out_format)
