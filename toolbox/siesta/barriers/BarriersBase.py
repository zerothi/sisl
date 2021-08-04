# -*- coding: utf-8 -*-
########################################################################################
# Copyright (c), The SiestaBarriers authors. All rights reserved.                       #
#                                                                                      #
# SiestaBarriers is hosted on GitHub at https://github.com/.................. #
# For further information on the license, see the LICENSE.txt file                     #
########################################################################################
#from __future__ import absolute_import

#from SiestaBarriers.Utils.utils_siesta import print_siesta_fdf,read_siesta_fdf

from .Utils.utils_siesta import print_siesta_fdf,read_siesta_fdf
#from Utils.utils_general import AtomIndex
#from SiestaBarriers.Utils.utils_siesta import print_siesta_fdf
from .Utils.utils_siesta import print_siesta_fdf
import sys
import inspect
pacakge_dir= inspect.getabsfile(print_siesta_fdf).split("utils_siesta.py")[0]+"flos/"
#pacakge_dir= inspect.getabsfile(print_siesta_fdf).split("neb_base.py")[0]+"flos/"


__author__ = "Arsalan Akhatar"
__copyright__ = "Copyright 2021, SIESTA Group"
__version__ = "0.1"
__maintainer__ = "Arsalan Akhtar"
__email__ = "arsalan_akhtar@outlook.com," #+ \" miguel.pruneda@icn2.cat "
__status__ = "Development"
__date__ = "Janurary 30, 2021"


class SiestaBarriersBase():
    """
    The base class to compute the different images for neb
    
    host_path                    : Path of Calculations
    host_structure               :
    initial_relaxed_path         :
    initial_structure            :
    final_relaxed_path           :
    final_structure              : 
    number_of_images             :
    image_direction              :
    neb_scheme                   :
    trace_atom_initial_position  :
    trace_atom_final_position    :
    kicked_atom_final_position   :
    switched_atom_final_position :
    ring_atoms_paths             :
    ghost                        :
    relaxed                      :
    ghost_info                   :
    """
    def __init__(self,
                 host_path = "./",
                 host_fdf_name = None,
                 host_structure = None ,
                 initial_relaxed_path = None,
                 initial_relaxed_fdf_name = None,
                 initial_structure = None,
                 final_relaxed_path = None,
                 final_relaxed_fdf_name =None,
                 final_structure = None,
                 pseudos_path = None,
                 trace_atom_initial_position = None,
                 trace_atom_final_position = None,
                 kicked_atom_final_position = None,
                 switched_atom_final_position = None,
                 ring_atoms_index = None,
                 ring_atoms_paths = None,
                 neb_results_path = None,
                 flos_path = None, 
                 flos_file_name_neb = 'neb.lua',
                 flos_file_name_relax = 'relax_geometry_lbfgs.lua',
                 relax_engine = 'lua',
                 interpolation_method = 'idpp',
                 exchange_direction = 'z' ,
                 number_of_images = None,
                 neb_scheme = 'vacancy-exchange',
                 ghost = False,
                 relaxed = False,
                 tolerance_radius = [1.0,1.0,1.0],
                 atol = 1e-2,
                 rtol = 1e-2,
                ):

        self.host_path = host_path
        self.host_fdf_name = host_fdf_name
        self.host_structure = host_structure
        
        self.initial_relaxed_path = initial_relaxed_path
        self.initial_relaxed_fdf_name = initial_relaxed_fdf_name
        self.initial_structure = initial_structure
        self.final_relaxed_path = final_relaxed_path
        self.final_relaxed_fdf_name = final_relaxed_fdf_name 
        self.final_structure = final_structure
        
        self.pseudos_path = pseudos_path


        self.trace_atom_initial_position = trace_atom_initial_position
        self.trace_atom_final_position = trace_atom_final_position
        self.kicked_atom_final_position = kicked_atom_final_position 
        self.switched_atom_final_position = switched_atom_final_position
        self.ring_atoms_index = ring_atoms_index
        self.ring_atoms_paths = ring_atoms_paths 
        self.neb_results_path = neb_results_path

        self.interpolation_method = interpolation_method
        self.exchange_direction = exchange_direction   
        self.number_of_images = number_of_images
        self.neb_scheme = neb_scheme
        self.ghost = ghost
        self.relaxed = relaxed 

        self.flos_path = pacakge_dir
        self.flos_file_name_neb = flos_file_name_neb
        self.flos_file_name_relax = flos_file_name_relax
        self.relax_engine = relax_engine

        self.tolerance_radius = tolerance_radius
        self.atol = atol
        self.rtol = rtol
        
        self.wellcome()
        self.setup()

    def wellcome(self):
        """
        """
        print ("---------------------------")
        print ("Wellcome To SiestaBarriers")
        print ("      Version : {}".format(__version__))
        print ("---------------------------")


    def setup(self):
        """
        Setup the workchain
        """

        print(" Check If NEB scheme is valid ...")
        neb_schemes_available = ["vacancy-exchange",
                                 "exchange",
                                 "interstitial",
                                 "kick",
                                 "switch",
                                 "ring",
                                 "manual"
                                        ]
        if self.neb_scheme is not None:
            if self.neb_scheme not in neb_schemes_available:
                print("NOT IMPLEMENTED")
                sys.exit()
            else:
                print("NEB image scheme is: {}".format(self.neb_scheme))
                #return self.neb_scheme
        if self.ghost == True:
            print(" NOTE: The Ghost Support Basis is True You Have To Provide The Basis Set Of Ghost Specie/s via PAO.Basisblock! ")
        else:
            print(" No Ghost Support Basis")

    def is_none_scheme(self):
        """
        Check if None correction scheme is being used
        """
        return self.inputs.correction_scheme == "none"


    def is_scheme_vacancy_exchange(self):
        """
        Check if NEB scheme vacancy_exchange is being used
        """
        return self.neb_scheme == "vacancy_exchange"

    def is_scheme_exchange(self):
        """
        Check if NEB scheme exchange is being used
        """
        return self.neb_scheme == "exchange"

    def is_scheme_interstitial(self):
        """
        Check if NEB scheme interstitial is being used
        """
        return self.neb_scheme == "interstitial"

    def is_scheme_kick(self):
        """
        Check if NEB scheme interstitial is being used
        """
        return self.neb_scheme == "kick"

    def is_scheme_switch(self):
        """
        Check if NEB scheme interstitial is being used
        """
        return self.neb_scheme == "switch"

    def is_scheme_ring(self):
        """
        Check if NEB scheme ring is being used
        """
        return self.neb_scheme == "ring"

    #---------------------------------------------------------
    # Set Methods
    #---------------------------------------------------------
    def set_host_path(self,host_path):
        """
        """
        self.host_path = host_path
    def set_trace_atoms_initial_final_position(self,trace_atom_initial_position,trace_atom_final_position):
       """
       """
       self.trace_atom_initial_position = trace_atom_initial_position
       self.trace_atom_final_position = trace_atom_final_position
    def set_initial_relaxed_path(self,initial_relaxed_path):
        """
        """
        self.initial_relaxed_path = initial_relaxed_path
    def set_final_relaxed_path(self,final_relaxed_path):
        """
        """
        self.final_relaxed_path = final_relaxed_path
    def set_host_structure(self,host_structure):
        """
        """
        self.host_structure = host_structure
    def set_initial_structure(self,initial_structure):
        """
        """
        self.initial_structure = initial_structure
    def set_final_structure(self,final_structure):
        """
        """
        self.final_structure = final_structure
    def set_exchange_direction(self,exchange_direction):
        """
        """
        self.exchange_direction = exchange_direction
    def set_kicked_atom_final_position(self,kicked_atom_final_position):
        """
        """
        self.kicked_atom_final_position = kicked_atom_final_position
    def set_switched_atom_final_position(self,switched_atom_final_position):
        """
        """
        self.switched_atom_final_position = switched_atom_final_position
    def set_ring_atoms_index(self,ring_atoms_index):
        """
        """
        self.ring_atoms_index = ring_atoms_index

    def set_ring_atoms_paths(self,ring_atoms_paths):
        """
        """
        self.ring_atoms_paths = ring_atoms_paths
    def set_interpolation_method(self,interpolation_method):
        """
        """
        self.interpolation_method = interpolation_method
    def set_number_of_images(self,number_of_images):
        """
        """
        self.number_of_images = number_of_images
    def set_neb_scheme(self,neb_scheme):
        """
        """
        self.neb_scheme = neb_scheme
    def set_ghost(self,ghost):
        """
        """
        if ghost == False and self.neb_scheme == "vacancy-exchange":
            print ("For Vacancy Exchange Be Carefull with Localized Basis Set Codes !!!")
        self.ghost = ghost
    def set_pseudos_path(self,pseudos_path):
        """
        """
        self.pseudos_path = pseudos_path
    def set_relaxed(self,relaxed):
        """
        """
        self.relaxed = relaxed
        if self.relaxed == True:
            print("########################################################")
            print("                       NOTE                             ")
            print(" You are setting relaxed flag to (True), You have to    ")
            print(" provide relaxed path & fdf name for both (endpoint)    ")
            print(" initial and final structures!                          ")
            print("########################################################")
   
    def set_flos_path(self,flos_path):
        """
        """
        self.flos_path = flos_path
    
    def set_flos_file_name_neb(self,flos_file_name_neb):
        """
        default for default name
        """
        if flos_file_name_neb == 'default':
            self.flos_file_name_neb = 'neb.lua'
        else:
            self.flos_file_name_neb = flos_file_name_neb

    def set_flos_file_name_relax(self,flos_file_name_relax):
        """
        default for default name
        """
        if flos_file_name_relax == 'default':
            self.flos_file_name_relax = 'relax_geometry_lbfgs.lua'
        else:
            self.flos_file_name_relax = flos_file_name_relax


    def set_neb_results_path(self,neb_results_path):
        """
        """
        self.neb_results_path = neb_results_path
    
    def set_initial_relaxed_fdf_name(self,initial_relaxed_fdf_name):
       """
       """
       self.initial_relaxed_fdf_name = initial_relaxed_fdf_name
   
    def set_final_relaxed_fdf_name(self,final_relaxed_fdf_name):
       """
       """
       self.final_relaxed_fdf_name = final_relaxed_fdf_name


   #===========================================================

