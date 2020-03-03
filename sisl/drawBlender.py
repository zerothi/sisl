
import bpy
import addon_utils
import numpy as np

def draw_structure(path):

    #Enable the atomic blender add-on, just in case it wasn't enabled
    addon_utils.enable("io_mesh_atomic")

    #Remove everything (there will probably be the default cube in the initial file)
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False, confirm=False)

    #Import the geometry
    bpy.ops.import_mesh.xyz(filepath=path, use_camera=False, use_lamp=False )

def lighting(coord_lims):

    [xmin, xmax], [ymin, ymax], [zmin, zmax] = coord_lims

    radius = np.max(abs(coord_lims))

    #Key
    bpy.ops.object.light_add(type='AREA', radius=radius, location=(0,0,zmax + 10))
    light = bpy.context.object
    light.data.cycles.max_bounces = 1
    light.data.energy = 5000

    #Fill
    bpy.ops.object.light_add(type='AREA', radius=radius, location=(0,0,zmin - 10), rotation = (3.14,0,0))
    light = bpy.context.object

    light.data.cycles.max_bounces = 1
    light.data.energy = 200

    #Sides
    bpy.ops.object.light_add(type='AREA', radius=radius, location=(0,ymax + 10,0), rotation = (-3.14/2,0,0))
    light = bpy.context.object

    light.data.cycles.max_bounces = 1
    light.data.energy = 200

    bpy.ops.object.light_add(type='AREA', radius=radius, location=(0,ymin - 10,0), rotation = (3.14/2,0,0))
    light = bpy.context.object

    light.data.cycles.max_bounces = 1
    light.data.energy = 200

def camera(coord_lims):

    [xmin, xmax], [ymin, ymax], [zmin, zmax] = coord_lims

    factor = 15

    #Create the camera and set it as the scene camera
    bpy.ops.object.camera_add(location=(0, 0, zmax + 10), 
                              rotation=(0, 0, 0))

    cam = bpy.context.object
    bpy.context.scene.camera = cam
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

def render():
    #Define the output path
    bpy.context.scene.render.filepath = '/home/pfebrer/image'

    #Define the renderer
    bpy.context.scene.render.engine = 'CYCLES'

    #And render :)
    bpy.ops.render.render('INVOKE_DEFAULT', write_still=True)

if __name__ == "__main__":
    
    draw_structure("/home/pfebrer/Simulations/water.xyz")
    lighting()
    camera()
    render()