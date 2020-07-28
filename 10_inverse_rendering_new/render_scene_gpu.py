import os
import time
import numpy as np
import enoki as ek

import mitsuba
mitsuba.set_variant('gpu_autodiff_rgb')

from mitsuba.core import xml, Thread, Transform4f, Bitmap, Float, Vector1f, Vector3f, UInt32
from mitsuba.python.util import traverse
from mitsuba.python.autodiff import render, write_bitmap, Adam

mitsuba_path = "/home/udemegane/mitsuba2_optix7/"
scene_folder = "exp_invert_vertex"
scene_name = "itr_sphere.xml"

# Convert flat array into a vector of arrays (will be included in next enoki release)
def ravel(buf, dim = 3):
    idx = dim * UInt32.arange(ek.slices(buf) // dim)
    return Vector3f(ek.gather(buf, idx), ek.gather(buf, idx + 1), ek.gather(buf, idx + 2))

# Return contiguous flattened array (will be included in next enoki release)
def unravel(source, target, dim = 3):
    idx = UInt32.arange(ek.slices(source))
    for i in range(dim):
        ek.scatter(target, source[i], dim * idx + i)

# Prepare output folder
output_path = "output/invert_vertex/"
if not os.path.isdir(output_path):
    os.makedirs(output_path)

# Load example scene
scene_folder = mitsuba_path + '/resources/data/docs/examples/' + scene_folder +"/"
Thread.thread().file_resolver().append(scene_folder)
scene = xml.load_file(scene_folder + scene_name)

params = traverse(scene)
positions_initial = ravel(params['object.vertex_positions_buf'])
normals_initial = ravel(params['object.vertex_normals_buf'])

# Create differential parameter to be optimized
vertex_count = ek.slices(positions_initial)
translate_ref = Float().zero(vertex_count)

# Create a new ParameterMap (or dict)
params_optim = {
    "translate" : translate_ref,
}

# Apply the transformation to mesh vertex position and update scene (e.g. Optix BVH)
def apply_limited_transformation(amplitude = 0.3):
    diff = Vector3f(normals_initial[0] * translate_ref,
                    normals_initial[1] * translate_ref,
                    normals_initial[2] * translate_ref)
    new_positions = diff * amplitude + positions_initial
    unravel(new_positions, params['object.vertex_positions_buf'])
    params.set_dirty('object.vertex_positions_buf')
    params.update()

# Render a reference image (no derivatives used yet)
#apply_limited_transformation()
image_ref = render(scene, spp=16)
crop_size = scene.sensors()[0].film().crop_size()
write_bitmap(output_path + 'out_ref_test2.exr', image_ref, crop_size)
print("Write " + output_path + "out_ref_test2.exr")