from scipy.spatial.transform import Rotation
import numpy as np
from math import acos
import os
from ._array import arrayd

try:
    import bpy
    from mathutils import Matrix, Vector

    try:
        import addon_utils
        INSIDE_BLENDER = True
        BLENDER_ASPACKAGE = False
        
    except ModuleNotFoundError:
        BLENDER_ASPACKAGE = True
        INSIDE_BLENDER = False

    BLENDER_AVAIL = True

except ModuleNotFoundError as BLENDER_IMPORT_EXCEPTION:
    BLENDER_IMPORT_EXCEPTION = BLENDER_IMPORT_EXCEPTION
    BLENDER_AVAIL = False
    BLENDER_ASPACKAGE = False
    INSIDE_BLENDER = False

__all__ = ['BlenderScene', 'blender']

class BlenderScene:

    """ Serves as an interface between sisl and blender.
    
    The `BlenderScene` class aims to help displaying sisl's
    scientifically meaningful classes in Blender (https://www.blender.org/)

    Any sisl class with a `__blender__()` method can be added to a
    scene by using the `àdd()` method.

    You can use this class directly or use the `sisl.blender` function,
    which will create an instance of this class.

    **IMPORTANT:** This class needs to import blender modules, therefore you
    need to be either running the script inside blender with sisl installed or
    have blender as a python module in your environment. See https://github.com/zerothi/sisl/pull/171
    for more info on how to achieve this.

    ~/myBlenderScript.py:
    .. code:: python
       import sisl
       geom = sisl.geom.fcc(3, "C")
       blender(square, camera_settings={}, lighting_settings={}, render_settings={})

    ```
    blender -P ~/myBlenderScript.py
    ```

    Attributes
    ----------
    ops: bpy.ops
    context: bpy.context
    data: bpy.data
    filepath: str
        the path where output files are saved by default.
    bounding_box: np.ndarray of shape (2,3)
        the limits of a box surrounding all objects in the scene.
        [[xmin, xmax], [ymin,ymax], [zmin,zmax]]
    objs: list
        contains all the objects in the scene.
    lights: list
        contains all the lights that are present in the scene.
    xmin
    xmax
    ymin
    ymax
    zmin
    zmax

    Parameters
    ----------
    objs : array_like, optional
        the objects that should be added to the scene on initialization.
        This is equivalent to adding them with the `add()` method
    filepath : str, optional
        the path that will be used as default to save files, without extension.

        If not provided, it is set to ~/sislImage.

        Note that if a filepath is provided in the render method it will 
        take preference over the value provided here.

    """

    _POVs = {
        '+z': (0,0,1),
        '-z': (0,0,-1),
        '+y': (0,1,0),
        '-y': (0,-1,0),
        '+x': (1,0,0),
        '-x': (-1,0,0),
    }

    def __init__(self, objs = [], filepath = None):

        #Define some attributes that will be helpful so that bpy does not need to be imported elsewhere
        self.ops = bpy.ops
        self.context = bpy.context
        self.data = bpy.data
        self.atomic_blender = lambda: addon_utils.enable("io_mesh_atomic")

        self.filepath = filepath or os.path.join(os.path.expanduser("~"), "sislImage")

        #Clean the scene of any objects that were there (e.g. the default cube)
        self.clean()

        self.objs = []
        self.lights = []

        #If objects were provided on initialization, add them to the scene
        for obj in objs:
            self.add(obj)
    
    def rotate_object(self, rot_mat, obj):
        '''
        Rotates an object according to a given rotation matrix.

        This code is taken directly from the atomic blender add-on 
        (https://github.com/blender/blender-addons/blob/master/io_mesh_atomic/xyz_import.py)

        Parameters
        ------
        rot_max: mathutils.Matrix.Rotation
            the rotation matrix
        obj: blender object
            the object that we want to rotate
        '''

        self.ops.object.select_all(action='DESELECT')
        obj.select_set(True)

        # Decompose world_matrix's components, and from them assemble 4x4 matrices.
        orig_loc, orig_rot, orig_scale = obj.matrix_world.decompose()

        orig_loc_mat   = Matrix.Translation(orig_loc)
        orig_rot_mat   = orig_rot.to_matrix().to_4x4()
        orig_scale_mat = (Matrix.Scale(orig_scale[0],4,(1,0,0)) @
                        Matrix.Scale(orig_scale[1],4,(0,1,0)) @
                        Matrix.Scale(orig_scale[2],4,(0,0,1)))

        # Assemble the new matrix.
        obj.matrix_world = orig_loc_mat @ rot_mat @ orig_rot_mat @ orig_scale_mat

    def update_bounding_box(self):
        '''
        Updates the bounding box of the model so that lights and cameras can be placed correctly.
        '''
        self.bounding_box = np.array([
            np.min([np.min(obj.bound_box, axis=0 ) for obj in self.data.objects], axis = 0),
            np.max([np.max(obj.bound_box, axis=0 ) for obj in self.data.objects], axis = 0),
        ]).T

        return self

    def get_limits(self, min_max = None, axis = [0,1,2]):

        '''
        Gets the limit in a given axis of the bounding box that surrounds the whole scene.

        It does not take into account lights and cameras.

        Parameters
        ---------
        min_max: {'min', 'max'}, optional
            specify which limit should be returned. Leave empty for both.
        axis: int or array-like, optional
            the axis, or axes, for which we want the limits

        Returns
        ---------
        float or numpy.ndarray:
            If only a single value is requested, a float is returned.
            If more than one value is requested an array is returned with the first dimension being
            the axis and the second one "min_max".
        ''' 

        second_dim = 0 if min_max == "min" else (1 if min_max == "max" else [0,1])

        return self.bounding_box[axis, second_dim]

    def min(self, axis = None):
        return self.get_limits('min', axis)

    def max(self, axis = None):
        return self.get_limits('max', axis)

    @property
    def xmin(self):
        return self.min(0)
    
    @property
    def xmax(self):
        return self.max(0)
    
    @property
    def ymin(self):
        return self.min(1)
    
    @property
    def ymax(self):
        return self.max(1)
    
    @property
    def zmin(self):
        return self.min(2)

    @property
    def zmax(self):
        return self.max(2)
    
    def bound_box_intersection(self, vec):
        '''
        Gets the point where a semi infinte vector in the direction of vec intersects with the bounding box

        Parameters
        -------
        vec: array-like of shape (3,)
            the direction along we want the limit of the cube. 
            A semi infinite vector in this direction and starting at (0,0,0) will be considered.
        '''
        vec = np.array(self._POVs[vec]) if isinstance(vec, str) else np.array(vec)

        #Get the predominant axis of the provided vector
        main_axis = np.argmax(abs(vec))

        #Get the value that defines the face of the cube that the vector is going to intersect
        get_limits = self.get_limits('min' if vec[main_axis] < 0 else 'max', main_axis)

        #Get the value of lambda in the 3D line parametric equation (Origin = (0,0,0))
        lambda_val = get_limits / vec[main_axis]

        return np.array([get_limits if axis == main_axis else comp*lambda_val for axis, comp in enumerate(vec)])

    def POV(self, POV, opposite=False):
        '''
        Returns the parameters that correspond to a given point of view.

        Parameters
        ------
        POV: {'+z', '-z', '-y', '+y', '-x', '+x'} or array-like
        opposite: boolean, optional
            set this to true if you want just the opposite point of view from the one you provided.
        '''

        loc_vec = np.array(self._POVs[POV]) if isinstance(POV, str) else np.array(POV)
        loc_vec *= -1 if opposite else 1
        camera_vec = [0,0,1] #This is the default vector along which camera points
        horizontal_camera_vec =  [1,0,0] #This is the original vector for the horizontal side

        #Calculate the angle and the rotation axis to get from camera_vec to loc_vec
        if (camera_vec == loc_vec).all():
            euler_rot = (0,0,0)
        elif (camera_vec == -loc_vec).all():
            euler_rot = (np.pi, 0, 0)
        else:
            angle = acos( np.dot(camera_vec, loc_vec) / (np.linalg.norm(camera_vec)*np.linalg.norm(loc_vec)) )
            rot_vec = np.cross(camera_vec, loc_vec)
            
            rot = Rotation.from_rotvec( rot_vec / np.linalg.norm(rot_vec) * angle)
            euler_rot = rot.as_euler('xyz')

        return {
            "direction": np.array(loc_vec),
            "rotation": euler_rot ,
            "horizontal": Rotation.from_euler("xyz", euler_rot).apply([1,0,0]) #This is the vector that ends up in the horizontal side
        }

    def clean(self):
        '''
        Cleans the scene by removing all objects from the 3d viewport.

        It does not delete the objects from the BlenderScene instance.
        For that, use the reset_*() method
        '''

        #Remove everything (there will probably be the default cube in the initial file)
        self.ops.object.select_all(action='SELECT')
        self.ops.object.delete(use_global=False, confirm=False)

        return self

    def add(self, obj):

        self.objs.append(obj)

        obj.__blender__(self)

        self.update_bounding_box()

        return self
    
    def light_add(self, *args, max_bounces=None, energy=None, color=None , **kwargs):

        self.ops.object.light_add(*args, **kwargs)

        light = self.context.object

        self.lights.append(light)

        light.data.cycles.max_bounces = max_bounces or light.data.cycles.max_bounces
        light.data.energy = energy or light.data.energy
        light.data.color = color or light.data.color

        return light

    def lighting(self, front="+z", front_intensity=2*10**8, front_distance=10, front_color=(1,1,1), fill_intensity=1*10**7, fill_distance=10, fill_color=(1,1,1) ):

        '''
        Sets the lighting

        Parameters
        --------
        front: {'+z', '-z', '-y', '+y', '-x', '+x'}, optional
            the direction where the light COMES FROM.
            This means that a value of '+z' will put the main light at the +z face of the scene, with light rays pointing
            to the center (light direction would be -z).
        front_intensity: int, optional
            the intensity of the main light.
        front_distance: float, optional
            how far is the main light from the outer face of the objects to be rendered.

            That is, if front is '+z' and front_distance is 15 and you have an object with a maximum z of 30, the light will
            be set at z=45.
        front_color: array-like of float (rgb format), optional
            the color of the main light in rgb format, where (0,0,0) is black and (1,1,1) is white.
        fill_intensity: int, optional
            the intensity of the filling lights.
            
            Filling lights are meant to avoid black areas, even at those areas that the main light can't reach. They simulate
            how light always bounces in real life making absolute dark a very rare thing.
        fill_distance: float, optional
            how far is the fill light from the outer face of the objects to be rendered.
        fill_color: array-like of float (rgb format), optional
            the color of the fill light in rgb format, where (0,0,0) is black and (1,1,1) is white.
        '''

        radius = np.max(self.bounding_box) - np.min(self.bounding_box)*2

        #Calculate the lights power according to intensity and radius
        front_power = front_intensity/radius**2
        fill_power = fill_intensity/radius**2
            
        max_bounces = 1

        #Key
        POV = self.POV(front)
        location = self.bound_box_intersection(POV["direction"]) + POV["direction"]*front_distance
        self.light_add(type='AREA', radius=radius, location=location, rotation=POV["rotation"], max_bounces = max_bounces, energy = front_power)
        
        #Fill
        POV = self.POV(front, opposite=True)
        location = self.bound_box_intersection(POV["direction"]) + POV["direction"]*fill_distance
        self.light_add(type='AREA', radius=radius, location=location, rotation=POV["rotation"],
            max_bounces = max_bounces, energy = fill_power)

        #Sides
        orto_axes = [ax for ax in ["x", "y", "z"] if ax not in front]
        for ax in orto_axes:
            for opposite in [True, False]:

                POV = self.POV(f'+{ax}', opposite=opposite)
                location = self.bound_box_intersection(POV["direction"]) + POV["direction"]*fill_distance

                self.light_add(type='AREA', radius=radius, location=location, rotation = POV["rotation"],
                    max_bounces = max_bounces, energy = fill_power)

    def camera(self, POV='+z', horizontal_axis=None, distance=None, resolution=(1920, 1080), location=None, rotation=(0,0,0), pointTo = None):

        '''
        Sets the camera for rendering

        Parameters
        --------
        POV: {'+z', '-z', '-y', '+y', '-x', '+x'}, optional
            the point of view from which the camera will capture images.
        horizontal_axis: {'+z', '-z', '-y', '+y', '-x', '+x'} or array-like of shape (3,), optional
            the axis that will correspond to the horizontal side of the picture

            If it is an array, it should be a vector that indicates the desired direction.

            If not provided, it is inferred from the POV like so:
                POV 'z' -> 'x'
                POV 'x' -> 'y'
                POV 'y' -> 'x'

        resolution: array-like of 2 int or int, optional
            the resolution in pixels of the horizontal and vertical sides of the image.

            If an integer is provided, a square image will be assumed.
        distance: float, optional
            how far is the camera from the outer face of the objects to be rendered.

            That is, if POV is '+z' and distance is 15 and you have an object with a maximum z of 30, the camera will
            be set at z=45.

            If not provided, it will be inferred to try to fit the object in the frame (NOT IMPLEMENTED YET, BUT VERY IMPORTANT)

        location: array-like of shape (3,), optional
            location of the image that will be directly passed to the camera.

            If this parameter is passed, it will have preference over POV and the rotation argument will be used.
        rotation: array-like of shape (3,), optional
            rotation of the camera in euler angles.

            It will only be used if location is provided. Otherwise, the rotation of the camera will be inferred from POV.
        pointTo: array-like of shape (3,), optional
            NOT IMPLEMENTED YET, DON'T USE
        '''

        #If no horizontal_axis is provided, we provide a default one depending on the POV
        if horizontal_axis is None:
            horizontal_axis = POV.replace("x", "y") if "x" in POV else POV.replace("y", "x") if "y" in POV else POV.replace("z", "x")

        #Define the resolution of the image
        if isinstance(resolution, int):
            resolution = [resolution]*2
        self.context.scene.render.resolution_x = resolution[0]
        self.context.scene.render.resolution_y = resolution[1]

        #Define the location and rotation of the camera
        if location is None:

            if distance is None:
                distance = 10 #Here distance should be calculated according to dimensions of bounding box

            POV = self.POV(POV)
            location = self.bound_box_intersection(POV["direction"]) + POV["direction"]*distance
            rotation = POV["rotation"]
            horizontal_axis = self._POVs[horizontal_axis] if isinstance(horizontal_axis, str) else horizontal_axis

        #Create the camera and set it as the scene camera
        self.ops.object.camera_add(location=location, 
                                rotation=rotation)

        cam = self.context.object
        self.context.scene.camera = cam

        #Rotate the camera along its axis to get the horizontal side that we want
        angle = Vector(horizontal_axis).angle(POV["horizontal"])
        self.rotate_object(Matrix.Rotation(angle, 4, POV["direction"]), cam)

        #cam.keyframe_insert("rotation_euler")
        #cam.keyframe_insert("location")
        
        #Rotate the camera

        # #Set the number of frames
        # bpy.context.scene.frame_end = 20

        # #
        # bpy.context.scene.frame_current = 10
        # cam.rotation_euler = (-3.14/2, 0, 0)
        # cam.location = (0,5,0)
        # cam.keyframe_insert("rotation_euler")
        # cam.keyframe_insert("location")

        # #
        # bpy.context.scene.frame_current = 20
        # cam.rotation_euler = (-3.14, 0, 0)
        # cam.location = (0,0,-5)
        # cam.keyframe_insert("rotation_euler")
        # cam.keyframe_insert("location")
    
    def render(self, filepath=None, engine='CYCLES', file_format='PNG', image_settings={}, transparent_bg=True, bg_color=(0.05,0.05,0.05,1),
        bg_strength=1.0 ,before_render=None):
        '''
        Renders the current scene

        Parameters
        ---------
        filepath: str, optional
            The path where the rendered image/movie should be stored, without extension
        engine: {'CYCLES', 'BLENDER_EEVEE', 'BLENDER_WORKBENCH' }, optional
            The render engine to be used. The default is 'CYCLES'.
            If you don't know the difference, you probably will be good with CYCLES, but you can try the other two.
            This parameter can take other values if you have external render engines in your blender install.
        file_format: str, optional
            The format to save the image (or video), in capital letters.
            You can see blender's GUI for all accepted values (In blender 2.8: Output properties -> Output -> File format).
            Some important ones are: 'PNG', 'JPEG', 'TIFF' (image) and 'AVI_JPEG', 'FFMPEG' (video)
        image_settings: dict, optional
            Dictionary that contains further image parameters, which are different for each format. 
            This is passed directly to self.context.scene.image_settings. Explore blender's GUI to know what you can tune, 
            or ignore this parameter to get the defaults.
        transparent_bg: bool, optional
            whether the background of the image should be transparent. Note that this is only possible with formats that
            support transparency, such as PNG.
        bg_color: array-like of float of shape (4,), optional
            background color in rgba format, where a stands for alpha, the opacity of the color.
            The color should be in the range (0,0,0,0) - (1,1,1,1).
        bg_strength: float, optional
            strength of the background color.
        before_render: function, optional
            A function that takes the BlenderScene object it's only argument and performs whatever operation it needs.

            This function is meant to avoid having endless parameters in the `render` function, since the tunable 
            possibilities are huge. Therefore the user can use this to set the configuration they need before rendering.

            Good practice would be that this function only modified render-related parameters, but do whatever you need.

            To understand how you can write this function, play with blender's GUI in scripting mode and you will see
            the python commands that your actions are triggering in the info panel.

            Note that you will find bpy.ops, bpy.context and bpy.data under BlenderScene.ops, BlenderScene.context and
            BlenderScene.data respectively. *You shouldn't need any blender imports inside your function*
        '''

        #Define the output path
        self.context.scene.render.filepath = filepath or self.filepath

        #Define the renderer
        self.context.scene.render.engine = engine

        image_settings = {"file_format": file_format, **image_settings}

        for key, val in image_settings.items():
            setattr(self.context.scene.render.image_settings, key, val)

        bpy.context.scene.render.film_transparent = transparent_bg
        bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[0].default_value = bg_color
        bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[1].default_value = bg_strength


        if callable(before_render):
            before_render(self)

        #And render :)
        self.ops.render.render('INVOKE_DEFAULT', write_still=True)

if BLENDER_AVAIL:
    def blender(obj, scene=None, lighting=True, lighting_settings={}, camera=True, camera_settings={}, render=True, render_settings={}):
        '''

        Creates a blender scene containing the given object and renders it if desired.

        Parameters
        ----------
        obj: any sisl object with a __blender__() method
            The object to be added to the 3D view.
        scene: BlenderScene, optional
            The scene in which to put the 3D representation of the object. If not provided, a new scene will be created.
        lighting: bool, optional
            Whether lighting should be set up or not.
        lighting_settings: dict, optional
            Dict whose values will be passed directly to `BlenderScene.lighting()`

            See the `BlenderScene.lighting` method to know what can be tuned.
        camera: bool, optional
            Whether the camera should be set up or not.
        camera_settings: dict, optional
            Dict whose values will be passed directly to `BlenderScene.camera()`

            See the `BlenderScene.camera` method to know what can be tuned.
        render: bool, optional
            Whether the scene should be rendered or not
        render_settings: dict, optional
            Dict whose values will be passed directly to `BlenderScene.render()`

            See the `BlenderScene.render` method to know what can be tuned.
        
        Returns
        --------
        BlenderScene:
            the BlenderScene object that has been created.
        
        '''
        if scene is None: 
            scene = BlenderScene()
            
        scene.add(obj)

        if lighting:
            scene.lighting(**lighting_settings)

        if camera:
            scene.camera(**camera_settings)

        if render:
            scene.render(**render_settings)

        return scene
else:
    def blender(obj, *args, **kwargs):
        raise BLENDER_IMPORT_EXCEPTION

# Clean up
del BLENDER_AVAIL
del INSIDE_BLENDER
del BLENDER_ASPACKAGE