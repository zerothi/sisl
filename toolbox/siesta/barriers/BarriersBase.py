# -*- coding: utf-8 -*-
########################################################################################
# Copyright (c), The SislSiestaBarriers authors. All rights reserved.                  #
# SislSiestaBarriers is hosted on GitHub at :                                          #
# https://github.com/zerothi/sisl/toolbox/siesta/barriers                              #
# For further information on the license, see the LICENSE.txt file                     #
########################################################################################
import pathlib
import shutil

from sisl import Geometry

from .Utils.utils_siesta import print_siesta_fdf

# You shouldn't use inspect, I assume __file__ would do just fine!
import inspect
pacakge_dir= inspect.getabsfile(print_siesta_fdf).split("utils_siesta.py")[0]+"flos/"


__author__ = "Arsalan Akhatar"
__copyright__ = "Copyright 2021, SIESTA Group"
__version__ = "0.1"
__maintainer__ = "Arsalan Akhtar"
__email__ = "arsalan_akhtar@outlook.com" 
__status__ = "Development"
__date__ = "Janurary 30, 2021"


class SiestaBarriersBase:

    # the documentation is not sphinx compatible. I would suggest
    # you read in the sisl code how to format the documentation.
    # Say read the sisl.geometry.py code
    # Also, the documention does not explain why the different parameters are
    # needed, for instance:
    #   host_structure               :  Sisl Structure Object
    # yes, it is a structure, but what does *host* mean? :)
    # Also, you should decide how much you want to use ase and how much
    # you want this to be an independent tool.
    # If you want this to rely on ASE, then perhaps it might be better to simply
    # use ase tools?

    
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
    # all these methods are never used, if they don't have a meaningful
    # usage, I would simply delete them.
    # If they are needed later, then they can be added.
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


class SiestaBarriersBaseNick:
    """
    The base class to compute the different images for neb
    
    Parameters
    ----------
    images : list of Geometry objects
       the images (including relaxed, first, and final image, last).
    path : callable or pathlib.Path or str
       path to write geometry to.
       If a callable it should have the following arguments ``image, index, total``
       where ``image`` corresponds to ``images[index]`` and ``total`` is the length of `images`.
       If a `pathlib.Path` it will be appended ``_{index}_{total}`` for checking indices.
       If a `str` it may contain formatting rules with ``{index}`` or ``{total}``.
       Internally ``self.path`` will be a callable with the above arguments.
    engine : handler for NEB calculation

    Examples
    --------

    I don't think you should implement this class to do the interpolation, what if the
    user wants to try some them-selves, it might be easier to provide wrappers for functions
    that interpolates, and then returns the full thing.
    It becomes much simpler and easier for the end user to fit their needs.

    Generally classes should do as little as possible, and preferably one thing only.
    Your classes are simply too difficult to understand for a new user. Is my bet.
    I am suggesting major changes here since I think that is required if this is to be used.
    Sorry to be this blunt.
    
    It also isn't clear to me the exact procedure or scope of these classes.
    Are you only focusing on using these scripts for the Lua engines? Or would
    you imagine later to extend with Python back-ends (say I-Pi?).

    I don't think you should have an IO class as well. It might be a set of
    functions, but otherwise it might be useful *in* the class if needed.

    """

    def __init__(self, images, path='image_{index}_{total}'):
        # Store all images in the class, convert to sisl.Geometry
        self.images = [Geometry.new(image) for image in images]
        # we need to have at least initial and final
        # While it doesn't make sense to calculate barriers for
        # two points, it might be useful for setting up initial and final
        # geometries in specific directories.
        assert len(self.images) >= 2

        if isinstance(path, str):
            path = pathlib.Path(path)

        if callable(path):
            def _path(image, index, total):
                return pathlib.Path(path(image, index, total))
        elif isinstance(path, pathlib.Path):
            # Convert to func
            def _path(image, index, total):
                return path.with_suffix(path.suffix + f"_{index}_{total}")
        else:
            raise ValueError("Unknown argument type for 'path' not one of [callable, str, Path]")
        self.path = _path

        # do you really need a welcome?
        # Users presumably know that they will use this script, and seeing this
        # everytime may be annoying and without any information.
        self.welcome()

    def __len__(self):
        """ Number of images (excluding initial and final) """
        return len(self.images) - 2

    @property
    def initial(self):
        return self.images[0]

    @property
    def final(self):
        return self.images[-1]

    def welcome(self):
        """
        """
        print ("---------------------------")
        print (" Welcome To SiestaBarriers ")
        print ("      Version : {}".format(__version__))
        print ("---------------------------")

    def _prepare_flags(self, files, overwrite):
        if isinstance(files, str):
            files = [files]
        if isinstance(overwrite, bool):
            # ensure a value per image
            overwrite = [overwrite for _ in range(total)]
        return files, overwrite

    def prepare(self, files='image.xyz', overwrite=False):
        """ Prepare the NEB calculation. This will create all the folders and geometries.

        Parameters
        ----------
        files : str or list of str
           write the image files in the ``self.path`` returned directory.
           If a list, all suffixes will be written.
           The directory will be created if not existing.
           The `files` argument may contain formatted strings ``{index}`` or
           ``{total}`` will be accessible.
        overwrite : bool or list of bool, optional
           whether to overwrite any existing files. Optionally a flag per image.
        """
        # generally you should try and avoid using os.chdir
        # It will cause you more pain than actual gain. ;)
        # Using relative paths are way more versatile and powerful
        total = len(self.images)
        files, overwrite = self._prepare_flags(files, overwrite)
        assert len(overwrite) == total

        for index, (overwrite, image) in enumerate(zip(overwrite, self.images)):
            path = self.path(image, index, total)
            path.mkdir(parents=True, exist_ok=True)
            # prepare the directory
            for file in files:
                file = path / file.format(index=index, total=total)
                if overwrite or not file.is_file():
                    # now write geometry
                    image.write(file)
                    

    # if you are not going to use the `set_*` methods, then don't add them.
    # If you have a set you should generally use them in your __init__ call
    # to ensure any setup is done correctly.
    # Also, there seemed to be quite a bit of inconsistency there.
    # I.e. you could setup everything, then change the number of images?
    # This does not make sense and it might be much better to have a simpler
    # class that is easier to maintain.


# The ManualNEB is simple the same as the base class (now)
# It may need some adjustments later.
class ManualNEB(SiestaBarriersBaseNick):
    pass


# It isn't clear at all how a user should use your scripts.
# Perhaps I should refrain from commenting more and you should
# ask a co-worker to run through it. I would highly suggest you
# make it *simpler* since it is too complicated to use.
# Possibly also ask Pol about some suggestions to make it simpler.

# You have lots of Utils.utils_*.py files.
# Instead, put everything that belongs to one method in 1 file.
# This is much simpler and is easier to figure out when things goes
# wrong.
# Also, you seem to have lots of duplicate code around? Why?
# I.e. utils_exchange.py and utils_interstitial.py?
# It really makes the code hard to follow ;)
# Could you also give examples of when the fractional vs. cartesian coordinates
# are useful? The way you check for fractional coordinates is not optimal, if useful at all.
# Why not force users to always use cartesian coordinates?

# I only think you should use CamelCase for classes.
# Methods should be lower_case_like_this (my opinion, makes it easier to
# remember method names).


class LuaNEB(SiestaBarriersBaseNick):
    # this class should implemnet the Lua stuff with copying files etc.
    def __init__(self, lua_scripts, images, *args, **kwargs):
        super().__init__(images, *args, **kwargs)
        if isinstance(lua_scripts, (str, Path)):
            lua_scripts = [lua_scripts]
        self.lua_scripts = [Path(lua_script) for lua_script in lua_scripts]

    def prepare(self, files='image.xyz', overwrite=False):
        super().prepare(files, overwrite)
        _, overwrite = self._prepare_flags(files, overwrite)

        for index, (overwrite, image) in enumerate(zip(overwrite, self.images)):
            path = self.path(image, index, total)
            for lua_script in self.lua_scripts:
                out = path / lua_script.name
                if overwrite or not out.is_file():
                    shutil.copy(lua_script, out)
    
