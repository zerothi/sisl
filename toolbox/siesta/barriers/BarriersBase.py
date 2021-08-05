# -*- coding: utf-8 -*-
########################################################################################
# Copyright (c), The SislSiestaBarriers authors. All rights reserved.                  #
# SislSiestaBarriers is hosted on GitHub at :                                          #
# https://github.com/zerothi/sisl/toolbox/siesta/barriers                              #
# For further information on the license, see the LICENSE.txt file                     #
########################################################################################

from .Utils.utils_siesta import print_siesta_fdf 
import inspect
pacakge_dir= inspect.getabsfile(print_siesta_fdf).split("utils_siesta.py")[0]+"flos/"


__author__ = "Arsalan Akhatar"
__copyright__ = "Copyright 2021, SIESTA Group"
__version__ = "0.1"
__maintainer__ = "Arsalan Akhtar"
__email__ = "arsalan_akhtar@outlook.com," 
__status__ = "Development"
__date__ = "Janurary 30, 2021"


class SiestaBarriersBase():
    """
    The base class to compute the different images for neb
    
    Inputs:
    --------------------------------------------------------------------------------------------------------------------------------
    
    host_structure               :  Sisl Structure Object
    initial_relaxed_path         :  Siesta relaxation Calculation for initial Configuration
    initial_relaxed_fdf_name     :  Siesta fdf name for initial Configuration
    initial_structure            :  Sisl Structure of initial Configuration
    final_relaxed_path           :  Siesta relaxation Calculation for final Configuration
    final_relaxed_fdf_name       :  Siesta fdf name for final Configuration
    final_structure              :  Sisl Structure of final configuration
    number_of_images             :  Number of images to be generate
    interpolation_method         :  The method of interpolation of images Linear Interpolation(li) or Image Dependent Pair Potential (idpp)
    exchange_direction           :  Direction of migration path for Exchange 
    
    Note: This is just for Exchange path
    
    tolerance_radius             :  Tolerance_radius threshold for Exchange path to not overlap the species
    trace_atom_initial_position  :  Index / Fractional Position / Cartesian Position , of Initial Specie to migrate
    trace_atom_final_position    :  Index / Fractional Position / Cartesian Position , of Final Specie to migrate 
    
    NOTE: in the case of interstitial there will be no Index option Since there is no specie in crystal in final configuration!)
    
    kicked_atom_final_position   : Index / Fractional Position / Cartesian Position of kicked Specie 
    ring_atoms_index             : Ring atoms index to specify which atoms are involve in ring path
    ring_atoms_paths             : Ring atoms path which indicate the path where the atoms moving 
    neb_results_path             : The Path of NEB calculations for Post-Processing 
    flos_path                    : Path to the flos directory for copying lua scripts the generated folders     
    flos_file_name_neb           : Name of neb lua script
    flos_file_name_relax         : Name of relaxing lua script
    relax_engine                 : Flag for relaxing using Siesta (CG) or LUA optimizer
    neb_scheme                   : NEB name string to check/pass/debug extra info 
    ghost                        : This is just for SIESTA or Codes with Localized Basis to have better descrition of basis especially when using VacancyExchange and Interstitial Case where there is no Basis in Initial/Final configuration
    relaxed                      : The Flag for checking the neb initial path generation is for unrelaxed or relaxed structures
    atol,rtol                    : threshould for finding specie via AtomIndex subroutine which takes the Frac/Cart coordinate and returns the index number of specie in the Geometry object array

    --------------------------------------------------------------------------------------------------------------------------------
    
    HOW it works:
            Each Barrier Type is a child class of SiestaBarriersBase & wil initialized with its own parameters ...
            in Most of cases user provide the host_structure sisl geometry object , and species index/position for the migration, the program will generate the initial and final configuration folders to relax, after running siesta or X code , user again pass the fdf name & path to results folder to generate the relaxed initial path for neb calculation and setup the folder for neb calculation...
            after running neb user could post-process the neb results via providing the neb result folder....
            

    HOW To USE :

    from toolbox.siesta.barriers import Manual
    import sisl
    initial= sisl.get_sile("./input-neg.fdf").read_geometry()
    final= sisl.get_sile("./input-pos.fdf").read_geometry()

    A = Manual(initial_structure=initial,
          final_structure=final,
          number_of_images=7,
          interpolation_method=''
              )
    A.Generate_Manual_Images()
    A.IO.Write_All_Images(folder_name='xsf',out_format='xsf')
    A.IO.Prepare_Endpoint_Relax()
    
    #########################
    ### AFter relaxation: ###
    #########################

    A.set_relaxed(True)
    A.set_initial_relaxed_path("/home/aakhtar/Calculations/2020/siesta/SiestaBarriers/Manual/new_domainwall/negative/results-fixing-cell/")
    A.set_final_relaxed_path("/home/aakhtar/Calculations/2020/siesta/SiestaBarriers/Manual/new_domainwall/positive/results-VC-Coor/")
    A.set_initial_relaxed_fdf_name("input.fdf")
    A.set_final_relaxed_fdf_name("input.fdf")

    A.Generate_Manual_Images()
    A.IO.Write_All_Images(folder_name='xsf',out_format='xsf')
    A.IO.prepare_NEB()

    ##################
    ### AFter NEB: ###
    ##################


    A.set_neb_results_path(PATH TO THE RESULT)
    A.NEB_Result()
    
    A.IO.Prepare_NEB_Analysis(.....)
        .Plot_NEB(....)
        .Write_n_NEB_Image(....)


    """
    


    def __init__(self,
                 host_structure = None ,
                 initial_relaxed_path = None,
                 initial_relaxed_fdf_name = None,
                 initial_structure = None,
                 final_relaxed_path = None,
                 final_relaxed_fdf_name =None,
                 final_structure = None,
                 number_of_images = None,
                 interpolation_method = 'idpp',
                 exchange_direction = 'z' ,
                 tolerance_radius = [1.0,1.0,1.0],
                 trace_atom_initial_position = None,
                 trace_atom_final_position = None,
                 kicked_atom_final_position = None,
                 ring_atoms_index = None,
                 ring_atoms_paths = None,
                 neb_results_path = None,
                 flos_path = None, 
                 flos_file_name_neb = 'neb.lua',
                 flos_file_name_relax = 'relax_geometry_lbfgs.lua',
                 relax_engine = 'lua',
                 neb_scheme = 'vacancy-exchange',
                 ghost = False,
                 relaxed = False,
                 atol = 1e-2,
                 rtol = 1e-2,
                ):

        self.host_structure = host_structure
        
        self.initial_relaxed_path = initial_relaxed_path
        self.initial_relaxed_fdf_name = initial_relaxed_fdf_name
        self.initial_structure = initial_structure
        self.final_relaxed_path = final_relaxed_path
        self.final_relaxed_fdf_name = final_relaxed_fdf_name 
        self.final_structure = final_structure
        self.number_of_images = number_of_images
        self.interpolation_method = interpolation_method
        self.exchange_direction = exchange_direction   
        self.tolerance_radius = tolerance_radius
        self.trace_atom_initial_position = trace_atom_initial_position
        self.trace_atom_final_position = trace_atom_final_position
        self.kicked_atom_final_position = kicked_atom_final_position 
        self.ring_atoms_index = ring_atoms_index
        self.ring_atoms_paths = ring_atoms_paths 
        self.neb_results_path = neb_results_path
        self.flos_path = pacakge_dir
        self.flos_file_name_neb = flos_file_name_neb
        self.flos_file_name_relax = flos_file_name_relax
        self.relax_engine = relax_engine
        self.neb_scheme = neb_scheme
        self.ghost = ghost
        self.relaxed = relaxed 
        self.atol = atol
        self.rtol = rtol
        
        self.welcome()
        self.setup()

    def welcome(self):
        """
        """
        print ("---------------------------")
        print (" Welcome To SiestaBarriers ")
        print ("      Version : {}".format(__version__))
        print ("---------------------------")


    def setup(self):
        """
        Setup the workchain
        """
        import sys

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

