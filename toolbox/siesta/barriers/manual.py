# -*- coding: utf-8 -*-
########################################################################################
# Copyright (c), The SislSiestaBarriers authors. All rights reserved.                  #
# SislSiestaBarriers is hosted on GitHub at :                                          #
# https://github.com/zerothi/sisl/toolbox/siesta/barriers                              #
# For further information on the license, see the LICENSE.txt file                     #
########################################################################################

from .BarriersBase import SiestaBarriersBase

class Manual(SiestaBarriersBase):
    """
    Init Object for Manual
    INPUTs:
           initial_structure = Sisl Stucture Object
           final_structure = Sisl Stucture Object
           number_of_images = # of images for NEB
           interpolation_method = idpp

    """
    def __init__(self,
                 initial_structure ,
                 final_structure ,
                 number_of_images,
                 ghost = True ,
                 interpolation_method = "idpp",
                 ):

        super().__init__(
                 relaxed = False,
                 initial_structure = initial_structure,
                 final_structure = final_structure,
                 number_of_images = number_of_images,
                 atol = 1e-2 ,
                 rtol = 1e-2,
                )

        self.initial_structure = initial_structure
        self.final_structure = final_structure
        self.interpolation_method = interpolation_method
        self.number_of_images = number_of_images 


    def Generate_Manual_Images(self):
        """

        """
        from .BarriersIO import SiestaBarriersIO
        import sisl
        from ase.neb import NEB 
        import sys
        from .Utils.utils_siesta import read_siesta_fdf,read_siesta_XV ,read_siesta_XV_before_relax

        if self.relaxed == True:
             print ("=================================================")
             print ("     The Relaxed Manual Image Generation ...     ")
             print ("=================================================")
             if self.initial_relaxed_path == None or self.final_relaxed_path == None :
                 sys.exit("intial/final relaxed path not provided")
             if self.initial_relaxed_fdf_name == None or self.final_relaxed_fdf_name == None :
                 sys.exit("intial/final relaxed fdf not provided")

             self.initial_structure = read_siesta_XV(self.initial_relaxed_path,self.initial_relaxed_fdf_name)["XV"]
             self.final_structure = read_siesta_XV(self.final_relaxed_path,self.final_relaxed_fdf_name)["XV"]
             initial = sisl.Geometry.toASE(self.initial_structure)
             final = sisl.Geometry.toASE(self.final_structure)
             #initial = read_siesta_XV(self.initial_relaxed_path,self.initial_relaxed_fdf_name)["XV"]
             #final = read_siesta_XV(self.final_relaxed_path,self.final_relaxed_fdf_name)["XV"]
 

        else:
             print ("=================================================")
             print ("       The Initial Manual Image Generation ...   ")
             print ("=================================================")
             initial = sisl.Geometry.toASE(self.initial_structure)
             final = sisl.Geometry.toASE(self.final_structure)
        
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
        self.neb = NEB(self.images)
        self.neb.interpolate(self.interpolation_method,mic=True)
        
 
        self.sisl_images = []
        for i in range(self.number_of_images+2):
            self.sisl_images.append(sisl.Geometry.fromASE(self.images[i]))
        
        self.IO = SiestaBarriersIO( neb_type = "manual",
                                    sisl_images = self.sisl_images,
                                    flos_path = self.flos_path,
                                    flos_file_name_relax = self.flos_file_name_relax,
                                    flos_file_name_neb = self.flos_file_name_neb,
                                    number_of_images = self.number_of_images,
                                    initial_relaxed_path = self.initial_relaxed_path,
                                    initial_relaxed_fdf_name = self.initial_relaxed_fdf_name,
                                    final_relaxed_path =  self.final_relaxed_path,
                                    final_relaxed_fdf_name =  self.final_relaxed_fdf_name,
                                    relax_engine = self.relax_engine,
                                    relaxed = self.relaxed,
                                    ghost = self.ghost
                                          )


    def NEB_Result(self):
        """
        """
        self.IO = SiestaBarriersIO( neb_type = "manual",
                                    sisl_images = self.sisl_images,
                                    flos_path = self.flos_path,
                                    flos_file_name_relax = self.flos_file_name_relax,
                                    flos_file_name_neb = self.flos_file_name_neb,
                                    number_of_images = self.number_of_images,
                                    initial_relaxed_path = self.initial_relaxed_path,
                                    initial_relaxed_fdf_name = self.initial_relaxed_fdf_name,
                                    final_relaxed_path =  self.final_relaxed_path,
                                    final_relaxed_fdf_name =  self.final_relaxed_fdf_name,
                                    relax_engine = self.relax_engine,
                                    relaxed = self.relaxed,
                                    ghost = self.ghost,
                                    neb_results_path = self.neb_results_path,
                                          )
