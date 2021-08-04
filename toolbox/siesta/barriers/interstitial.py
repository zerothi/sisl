# -*- coding: utf-8 -*-
########################################################################################
# Copyright (c), The SiestaBarriers authors. All rights reserved.                       #
#                                                                                      #
# SiestaBarriers is hosted on GitHub at https://github.com/.................. #
# For further information on the license, see the LICENSE.txt file                     #
########################################################################################
#from __future__ import absolute_import

from .BarriersBase import SiestaBarriersBase
import logging
__author__ = "Arsalan Akhatar"
__copyright__ = "Copyright 2021, SIESTA Group"
__version__ = "0.1"
__maintainer__ = "Arsalan Akhtar"
__email__ = "arsalan_akhtar@outlook.com," + \
        " miguel.pruneda@icn2.cat "
__status__ = "Development"
__date__ = "Janurary 30, 2021"

_log = logging.getLogger("toolbox.siesta.barrires.vacancy_exchange")

class Interstitial(SiestaBarriersBase):
    """
    Init Object for VacancyExchange
    INPUTs:
           host_structure = Sisl Stucture Object
           number_of_images = # of images for NEB
           trace_atom_initial_position = initial Atom Position/Index
           trace_atom_final_position = final Atom Position/Index
           interpolation_method = idpp
    """
    
    def __init__(self,
                 host_structure ,
                 number_of_images  ,
                 trace_atom_initial_position ,
                 trace_atom_final_position ,
                 interpolation_method = 'idpp',
                 ghost = True ,
                 ):
   
        super().__init__(
                 neb_scheme = 'interstitial',
                 relaxed = False,
                 host_structure = host_structure ,
                 number_of_images = number_of_images,
                 initial_relaxed_path = None,
                 initial_relaxed_fdf_name = None ,
                 final_relaxed_path = None ,
                 final_relaxed_fdf_name = None,
                 trace_atom_initial_position = trace_atom_initial_position ,
                 trace_atom_final_position = trace_atom_final_position ,
                 atol = 1e-2,
                 rtol = 1e-2,
                )
        self.host_structure = host_structure
        self.number_of_images = number_of_images
        self.interpolation_method = interpolation_method 
        self.ghost = ghost

    #---------------------------------------------------------
    # Main Methods
    #---------------------------------------------------------
 
    def Generate_Interstitial_Images(self):
        """

        """
        from .Utils.utils_siesta import read_siesta_fdf,read_siesta_XV ,read_siesta_XV_before_relax, FixingSislImages
        from .Utils.utils_interstitial import pre_prepare_sisl ,is_frac_or_cart_or_index,pre_prepare_ase_after_relax
        import os,sys 
        import glob,shutil
        from .BarriersIO import SiestaBarriersIO


        import sisl
        from ase.neb import NEB 
        if self.relaxed == True:
             print ("=================================================")
             print ("The Relaxed Interstitial Image Generation ...")
             print ("=================================================")

             if self.initial_relaxed_path == None or self.final_relaxed_path == None :
                 sys.exit("intial/final relaxed path not provided")
             if self.initial_relaxed_fdf_name == None or self.final_relaxed_fdf_name == None :
                 sys.exit("intial/final relaxed fdf not provided")

             #self.host_structure = read_siesta_fdf(self.host_path,self.host_fdf_name)
             self.initial_structure = read_siesta_XV(self.initial_relaxed_path,self.initial_relaxed_fdf_name)
             self.final_structure = read_siesta_XV(self.final_relaxed_path,self.final_relaxed_fdf_name)
             
             self.check_ghost()
             self.test = pre_prepare_ase_after_relax(self.initial_structure['XV'],self.final_structure['XV'],self.ghost)
             initial = sisl.Geometry.toASE(self.test['initial'])
             final = sisl.Geometry.toASE(self.test['final'])


        else:
             print ("=================================================")
             print ("The Initial Interstitial Image Generation ...")
             print ("=================================================")
             #if "fdf" in self.host_fdf_name:
             #    self.host_structure = read_siesta_fdf(self.host_path,self.host_fdf_name)['Geometry']
             #if "XV" in self.host_fdf_name:
             #    self.host_structure = read_siesta_XV_before_relax(self.host_path,self.host_fdf_name)['XV']
             #self.host_structure
             frac_or_cart_or_index = is_frac_or_cart_or_index(self.trace_atom_initial_position )
             self.test = pre_prepare_sisl(frac_or_cart_or_index,
                                     #self.host_structure['Geometry'],
                                     self.host_structure,
                                     self.trace_atom_initial_position , 
                                     self.trace_atom_final_position,
                                     self.rtol,
                                     self.atol,
                                     self.ghost
                                     )
             initial = sisl.Geometry.toASE(self.test['initial'])
             final = sisl.Geometry.toASE(self.test['final'])


        #%%*****************************************************************
        #%% Do the Image Creation with ASE Here
        #%%*****************************************************************
        if self.relaxed == True:
            print("NEB Interpolation for : Relaxed Structures with {}".format(self.interpolation_method))
        else:
            print("NEB Interpolation for : UnRelaxed Structures with {}".format(self.interpolation_method))
        self.images = [initial]
        for i in range(self.number_of_images):
            print ("Copying ASE For NEB Image ",i)
            self.images.append(initial.copy())
        self.images.append(final)
        print ("Copying ASE For NEB Image ",i+1)
        #%% 
        self.neb = NEB(self.images)
        if self.interpolation_method == 'li':
            #self.neb.interpolate(self.interpolation_method)
            print("DEBUG: NEB LI INTERPOLATED!")
            self.neb.interpolate(mic=True)
        if self.interpolation_method == 'idpp':
            print("DEBUG: NEB IDPP INTERPOLATED!")
            self.neb.interpolate('idpp',mic=True)
 
        
        self.sisl_images = []
        for i in range(self.number_of_images+2):
            #self.sisl_images.append(sisl.Geometry.fromASE(self.images[i]))
            temp = sisl.Geometry.fromASE(self.images[i])
            self.sisl_images.append(FixingSislImages(self.test['initial'], temp,"ghost",self.relaxed,'moving'))

        
        if self.ghost == True:
            print(" Putting Ghost in Sisl Geometry Object ")
            for i in range(self.number_of_images+2):
                self.sisl_images[i] = self.sisl_images[i].add(self.test['Ghost_initial'])
                self.sisl_images[i] = self.sisl_images[i].add(self.test['Ghost_final'])

                
        self.IO = SiestaBarriersIO(neb_type = 'vacancy-exchange',
                                   sisl_images = self.sisl_images,
                                   flos_path = self.flos_path,
                                   flos_file_name_relax = self.flos_file_name_relax,
                                   flos_file_name_neb = self.flos_file_name_neb,
                                   number_of_images = self.number_of_images,
                                   initial_relaxed_path = self.initial_relaxed_path,
                                   final_relaxed_path = self.final_relaxed_path,
                                   final_relaxed_fdf_name = self.final_relaxed_fdf_name,
                                   initial_relaxed_fdf_name = self.final_relaxed_path,
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

