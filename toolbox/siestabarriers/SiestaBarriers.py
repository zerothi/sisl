# -*- coding: utf-8 -*-
########################################################################################
# Copyright (c), The SiestaBarriers authors. All rights reserved.                       #
#                                                                                      #
# SiestaBarriers is hosted on GitHub at https://github.com/.................. #
# For further information on the license, see the LICENSE.txt file                     #
########################################################################################

from __future__ import absolute_import
from .SiestaBarriersBase import  SiestaBarriersBase
from .barriers.vacancy_exchange  import VacancyExchange
from .barriers.exchange import Exchange
#from .barriers.kick import Kick
#from .barriers.ring import Ring
#from .barriers.manual import Manual


#import Utils.utils_general

#import .Utils.utils_siesta

from .Utils.utils_siesta import *
#import SiestaBarriers.Utils.utils_siesta


__author__ = "Arsalan Akhatar"
__copyright__ = "Copyright 2021, SIESTA Group"
__version__ = "0.1"
__maintainer__ = "Arsalan Akhtar"
__email__ = "arsalan_akhtar@outlook.com," #+ \" miguel.pruneda@icn2.cat "
__status__ = "Development"
__date__ = "Janurary 30, 2021"


class SiestaBarriers(SiestaBarriersBase):
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
                 image_direction = None ,
                 trace_atom_initial_position = None,
                 trace_atom_final_position = None,
                 kicked_atom_final_position = None,
                 switched_atom_final_position = None,
                 ring_atoms_paths = None,
                 number_of_images = 7,
                 flos_path = None,
                 flos_file_name_neb = "neb.lua",
                 flos_file_name_relax = None,
                 neb_results_path = None,
                 relax_engine = None,
                 ghost = False,
                 relaxed = False,
                 neb_scheme =  "vacancy-exchange",
                 interpolation_method = "idpp",
                 tolerance_radius =  [1.0,1.0,1.0]

                 ):
        
        super().__init__(
                         host_path ,
                         host_fdf_name ,
                         host_structure ,
                         initial_relaxed_path ,
                         initial_relaxed_fdf_name ,
                         initial_structure ,
                         final_relaxed_path ,
                         final_relaxed_fdf_name ,
                         final_structure ,
                         image_direction ,
                         trace_atom_initial_position ,
                         trace_atom_final_position ,
                         kicked_atom_final_position ,
                         switched_atom_final_position ,
                         ring_atoms_paths ,
                         number_of_images ,
                         neb_scheme ,
                         flos_path ,
                         ghost ,
                         relaxed ,
                         flos_file_name_neb,
                         flos_file_name_relax,
                         neb_results_path,
                         relax_engine,
                         interpolation_method,
                         tolerance_radius,

                         )
        

        self.host_path = host_path
        print(self.host_path)
        self.host_fdf_name = host_fdf_name
        self.host_structure = host_structure
        self.initial_relaxed_path = initial_relaxed_path
        self.initial_relaxed_fdf_name = initial_relaxed_fdf_name
        self.initial_structure = initial_structure
        self.final_relaxed_path = final_relaxed_path
        self.final_relaxed_fdf_name = final_relaxed_fdf_name
        self.final_structure = final_structure
        self.image_direction = image_direction
        self.trace_atom_initial_position = trace_atom_initial_position
        self.trace_atom_final_position = trace_atom_final_position
        self.kicked_atom_final_position = kicked_atom_final_position
        self.switched_atom_final_position = switched_atom_final_position
        self.ring_atoms_paths = ring_atoms_paths
        self.number_of_images = number_of_images
        print(self.number_of_images)
        self.neb_scheme = neb_scheme
        self.flos_file_name_neb = flos_file_name_neb 
        self.interpolation_method = interpolation_method
        self.ghost = ghost
        #self.flos_path = flos_path
        self.tolerance_radius = tolerance_radius
       
        #super(SiestaBarriersBase,self).__init__()
        #super().__init__()
        
        

        print ("---------------------------")
        print ("Wellcome To SiestaBarriers")
        print ("      Version : {}".format(__version__))
        print ("---------------------------")
        
        self.setup() 

        if self.neb_scheme == "vacancy-exchange":
            print("The SiestaBarriers Setup for : {}".format(self.neb_scheme))
            self.VacancyExchange = VacancyExchange (host_path = self.host_path,
                                                    host_fdf_name= self.host_fdf_name,
                                                    host_structure = self.host_structure,
                                                    initial_relaxed_path = self.initial_relaxed_path,
                                                    initial_relaxed_fdf_name = self.initial_relaxed_fdf_name,
                                                    initial_structure = self.initial_structure,
                                                    final_relaxed_path = self.final_relaxed_path,
                                                    final_relaxed_fdf_name = self.final_relaxed_fdf_name,
                                                    final_structure = self.final_structure,
                                                    flos_path = self.flos_path,
                                                    interpolation_method = self.interpolation_method,
                                                    ghost = self.ghost,
                                                    trace_atom_initial_position = self.trace_atom_initial_position,
                                                    trace_atom_final_position = self.trace_atom_final_position,
                                                    number_of_images = self.number_of_images
                                                    )
        if self.neb_scheme == "exchange":
            print("The SiestaBarriers Setup for : {}".format(self.neb_scheme))
            self.Exchange = Exchange (host_path = self.host_path,
                                                    host_fdf_name= self.host_fdf_name,
                                                    host_structure = self.host_structure,
                                                    initial_relaxed_path = self.initial_relaxed_path,
                                                    initial_relaxed_fdf_name = self.initial_relaxed_fdf_name,
                                                    initial_structure = self.initial_structure,
                                                    final_relaxed_path = self.final_relaxed_path,
                                                    final_relaxed_fdf_name = self.final_relaxed_fdf_name,
                                                    final_structure = self.final_structure,
                                                    flos_path = self.flos_path,
                                                    trace_atom_initial_position = self.trace_atom_initial_position,
                                                    trace_atom_final_position = self.trace_atom_final_position,
                                                    tolerance_radius = self.tolerance_radius,
                                                    number_of_images = self.number_of_images
                                                    )
        if self.neb_scheme == "kick":
            print("The SiestaBarriers Setup for : {}".format(self.neb_scheme))
            self.Kick = Kick (host_path = self.host_path,
                                                    host_fdf_name= self.host_fdf_name,
                                                    host_structure = self.host_structure,
                                                    initial_relaxed_path = self.initial_relaxed_path,
                                                    initial_relaxed_fdf_name = self.initial_relaxed_fdf_name,
                                                    initial_structure = self.initial_structure,
                                                    final_relaxed_path = self.final_relaxed_path,
                                                    final_relaxed_fdf_name = self.final_relaxed_fdf_name,
                                                    final_structure = self.final_structure,
                                                    flos_path = self.flos_path,
                                                    trace_atom_initial_position = self.trace_atom_initial_position,
                                                    trace_atom_final_position = self.trace_atom_final_position,
                                                    kicked_atom_final_position = self.kicked_atom_final_position,
                                                    )

    
        if self.neb_scheme == "ring":
            print("The SiestaBarriers Setup for : {}".format(self.neb_scheme))
        
            self.Ring = Ring (host_path = self.host_path,
                                                    host_fdf_name= self.host_fdf_name,
                                                    host_structure = self.host_structure,
                                                    initial_relaxed_path = self.initial_relaxed_path,
                                                    initial_relaxed_fdf_name = self.initial_relaxed_fdf_name,
                                                    initial_structure = self.initial_structure,
                                                    final_relaxed_path = self.final_relaxed_path,
                                                    final_relaxed_fdf_name = self.final_relaxed_fdf_name,
                                                    final_structure = self.final_structure,
                                                    flos_path = self.flos_path,
                                                    ring_atoms_paths = self.ring_atoms_paths,
                                                    ring_atoms_index = self.ring_atoms_index
                                                    )

        if self.neb_scheme == "manual":
            print("The SiestaBarriers Setup for : {}".format(self.neb_scheme))
            self.Manual = Manual (#host_path = self.host_path,
                                                    #host_fdf_name= self.host_fdf_name,
                                                    #host_structure = self.host_structure,
                                                    interpolation_method = self.interpolation_method,
                                                    initial_relaxed_path = self.initial_relaxed_path,
                                                    initial_relaxed_fdf_name = self.initial_relaxed_fdf_name,
                                                    initial_structure = self.initial_structure,
                                                    final_relaxed_path = self.final_relaxed_path,
                                                    final_relaxed_fdf_name = self.final_relaxed_fdf_name,
                                                    final_structure = self.final_structure,
                                                    flos_path = self.flos_path
                                                    )

    #---------------------------------------------------------
    # Set Methods Overwrite the barriers method   
    #---------------------------------------------------------
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
 

    #================================================================
    #
    #================================================================
        

