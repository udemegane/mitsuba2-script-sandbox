import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__),'..'))
import localpath as lp
import time
import numpy as np
import enoki as ek

import mitsuba
mitsuba.set_variant('gpu_autodiff_rgb')

from mitsuba.core import xml, Thread, Transform4f, Bitmap, Float, Vector1f, Vector3f, UInt32, Struct
from mitsuba.render import SurfaceInteraction3f
from mitsuba.python.util import traverse
from mitsuba.python.autodiff import render, write_bitmap, Adam

mitsuba_path = "/home/udemegane/mitsuba2_optix7/"
img_ref_name = "out_ref.exr"

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
output_path = lp.output_path_base + os.path.basename(__file__)

if not os.path.isdir(output_path):
    os.makedirs(output_path)

# Load scene
scene_folder = mitsuba_path + '/resources/data/docs/examples/exp_invert_vertex/'
Thread.thread().file_resolver().append(scene_folder)
scene = xml.load_file(scene_folder + 'itr_sphere.xml')
scene_ref = xml.load_file(scene_folder + 'exp_test1.xml')

# Load a reference image (no derivatives used yet)
crop_size = scene.sensors()[0].film().crop_size()
bitmap_tmp = Bitmap(output_path + img_ref_name, Bitmap.FileFormat.OpenEXR).convert(Bitmap.PixelFormat.RGB,
                                                                                   Struct.Type.Float32,
                                                                                   srgb_gamma=False)
image_ref = np.array(bitmap_tmp).flatten()
image_ref = render(scene_ref, spp=6)
crop_size_ref = scene_ref.sensors()[0].film().crop_size()
write_bitmap(output_path + 'out_ref.exr', image_ref, crop_size_ref)
print("Write " + output_path + "out_ref.exr")

params = traverse(scene)
print(params)
positions_initial = ravel(params['object.vertex_positions_buf'])
normals_initial = ravel(params['object.vertex_normals_buf'])
vertex_count = params['object.vertex_count']

print(normals_initial)
# Create differential parameter to be optimized
translate_ref = Float().full(0.1, vertex_count)
#print(translate_ref * normals_initial[0])

# Create a new ParameterMap (or dict)
params_optim = {
    "translate" : translate_ref,
}
# Construct an Adam optimizer that will adjust the translation parameters
opt = Adam(params_optim, lr=0.02)

# Apply the transformation to mesh vertex position and update scene (e.g. Optix BVH)
def apply_limited_transformation(amplitude = 0.5):
    diff = Vector3f(normals_initial[0] * params_optim["translate"],
                    normals_initial[1] * params_optim["translate"],
                    normals_initial[2] * params_optim["translate"])
    print(params_optim)
    new_positions = diff * amplitude + positions_initial
    unravel(new_positions, params['object.vertex_positions_buf'])
    params.set_dirty('object.vertex_positions_buf')
    params.update()

apply_limited_transformation()

time_a = time.time()

iterations = 300
for it in range(iterations):
    # Perform a differentiable rendering of the scene
    image = render(scene,
                   optimizer=opt,
                   spp=2,
                   unbiased=True,
                   pre_render_callback=apply_limited_transformation)

    write_bitmap(output_path + 'out_%03i.exr' % it, image, crop_size)

    # Objective: MSE between 'image' and 'image_ref'
    ob_val = ek.hsum(ek.sqr(image - image_ref)) / len(image)

    # Back-propagate errors to input parameters
    ek.backward(ob_val)

    # Optimizer: take a gradient step -> update displacement map
    opt.step()

    print('Iteration %03i: error=%g' % (it, ob_val[0]), end='\r')

time_b = time.time()

print('%f ms per iteration' % (((time_b - time_a) * 1000) / iterations))