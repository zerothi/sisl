import numpy as np

from ._configurable import Configurable, afterSettingsInit, afterSettingsUpdate

try:
    import bpy

    try:
        import addon_utils
        INSIDE_BLENDER = True
        BLENDER_ASPACKAGE = False
        
    except ModuleNotFoundError:
        BLENDER_ASPACKAGE = True
        INSIDE_BLENDER = False

    BLENDER_AVAIL = True

except ModuleNotFoundError as BLENDER_IMPORT_EXCEPTION:
    BLENDER_AVAIL = False
    BLENDER_ASPACKAGE = False
    INSIDE_BLENDER = False

__all__ = ['BlenderScene', 'blender']

class BlenderScene(Configurable):

    _parameters = (

        {
            "key": "cameraDistance",
            "default": 10
        }

    ,)

    @afterSettingsInit
    def __init__(self, objs = []):

        #Define some attributes that will be helpful so that bpy does not need to be imported elsewhere
        self.ops = bpy.ops
        self.context = bpy.context
        self.data = bpy.data
        self.atomic_blender = lambda: addon_utils.enable("io_mesh_atomic")

        self.filepath = "/home/pfebrer/image"

        #Clean the scene of any objects that were there (e.g. the default cube)
        self.clean()

        self.objs = []
        self.lights = []

        #If objects were provided on initialization, add them to the scene
        for obj in objs:
            self.add(obj)

    def lim(self, min_max = None, axis = None):

        first_dim = [0,1] if axis is None else axis

        second_dim = 0 if min_max == "min" else (1 if min_max == "max" else [0,1])

        return self.bounding_box[first_dim, second_dim]

    def min(self, axis = None):
        return self.lim('min', axis)

    def max(self, axis = None):
        return self.lim('max', axis)
    
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
    
    def clear(self):

        self.objs = []

        return self.clean()

    def clean(self):

        #Remove everything (there will probably be the default cube in the initial file)
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete(use_global=False, confirm=False)

        return self

    def add(self, obj):

        self.objs.append(obj)

        self.bounding_box = obj.__blender__(self)

        return self
    
    def light_add(self, *args, max_bounces = None, energy = None , **kwargs):

        self.ops.object.light_add(*args, **kwargs)

        light = self.context.object

        self.lights.append(light)

        light.data.cycles.max_bounces = max_bounces or light.data.cycles.max_bounces
        light.data.energy = energy or light.data.energy

        return light

    @afterSettingsUpdate
    def lighting(self):

        radius = np.max(abs(self.bounding_box))

        max_bounces = 1000
        key_power = 5000
        key_distance = 10

        fill_power = 200
        fill_distance = 10

        #Key
        self.light_add(type='AREA', radius=radius, location=(0,0, self.zmax + key_distance), max_bounces = max_bounces, energy = key_power)
        
        #Fill
        self.light_add(type='AREA', radius=radius, location=(0,0, self.zmin - fill_distance), rotation = (np.pi,0,0),
            max_bounces = max_bounces, energy = fill_power)

        #Sides
        for sign, yref in zip([-1, 1], [self.ymin, self.ymax]):

            self.light_add(type='AREA', radius=radius, location=(0,yref + sign*fill_distance,0), rotation = (sign*np.pi/2,0,0),
                max_bounces = max_bounces, energy = fill_power)

    @afterSettingsUpdate
    def camera(self):

        #Create the camera and set it as the scene camera
        self.ops.object.camera_add(location=(0, 0, self.zmax + 10), 
                                rotation=(0, 0, 0))

        cam = self.context.object
        self.context.scene.camera = cam
        cam.keyframe_insert("rotation_euler")
        cam.keyframe_insert("location")
        
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
    
    @afterSettingsUpdate
    def render(self):
        #Define the output path
        self.context.scene.render.filepath = self.filepath

        #Define the renderer
        self.context.scene.render.engine = 'CYCLES'

        #And render :)
        self.ops.render.render('INVOKE_DEFAULT', write_still=True)

if BLENDER_AVAIL:
    def blender(obj, scene=None, lighting=True, camera=True, render=True):
        '''

        Creates a blender scene containing the given object and renders it if desired.

        Parameters
        ----------
        obj: any sisl object with a __blender__() method
            The object to be added to the 3D view.
        scene: BlenderScene, optional
            The scene in which to put the 3D representation of the object. If not provided, a new scene will be created.
        lighting: bool, optional
            Whether lighting should be setup
        render: bool, optional
            Whether the scene should be rendered or not
        
        
        '''
        if scene is None: 
            scene = BlenderScene()
            
        scene.add(obj)

        if lighting:
            scene.lighting()

        if camera:
            scene.camera()

        if render:
            scene.render()

        return scene
else:
    def blender(obj, *args, **kwargs):
        raise BLENDER_IMPORT_EXCEPTION

# Clean up
del BLENDER_AVAIL
del INSIDE_BLENDER
del BLENDER_ASPACKAGE

if __name__ == "__main__":
    
    v = BlenderScene()
    v.draw_structure("/home/pfebrer/Simulations/water.xyz")
    v.lighting()
    v.camera()
    v.render()