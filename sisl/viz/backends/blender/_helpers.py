import bpy


def delete_all_objects():
    """Deletes all objects present in the scene"""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False, confirm=False)
