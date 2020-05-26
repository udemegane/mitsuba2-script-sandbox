import enoki as ek
import mitsuba
mitsuba.set_variant('gpu_autodiff_rgb')

from mitsuba.core import Thread, Color3f, Vector3f, Point3f
from mitsuba.core.xml import load_file
from mitsuba.python.util import traverse
from mitsuba.python.autodiff import render, write_bitmap, Adam


# Load the Cornell Box
Thread.thread().file_resolver().append('cbox')
scene = load_file('cbox/cbox.xml')

# Find differentiable scene parameters
params = traverse(scene)
print(params)
#print(params['OBJMesh.vertex_positions'])

#
# convert largebox to smallbox
param_ref = Point3f(params['OBJMesh_8.vertex_positions'])
param_ini = Point3f(params['OBJMesh_7.vertex_positions'])
params.keep(['OBJMesh_8.vertex_positions'])
print(param_ref)
print(param_ini)

# change box location
params['OBJMesh_8.vertex_positions'] = param_ini
params.update()

# Render a reference image (no derivatives used yet)
image_ref = render(scene, spp=8)
crop_size = scene.sensors()[0].film().crop_size()
write_bitmap('out_ini.png', image_ref, crop_size)

params['OBJMesh_8.vertex_positions'] = param_ref
params.update()


opt = Adam(params, lr=5)

iterations = 100
for it in range(iterations):
    image = render(scene, optimizer=opt, unbiased=True, spp=8)
    write_bitmap('out/out_%03i.png' % it, image, crop_size)

    # Objective: MSE between 'image' and 'image_ref'
    ob_val = ek.hsum(ek.sqr(image - image_ref)) / len(image)

    # Back-propagate errors to input parameters
    ek.backward(ob_val)

    # Optimizer: take a gradient step
    opt.step()

    # Compare iterate against ground-truth value
    err_ref = ek.hsum(ek.sqr(param_ref - params['OBJMesh_8.vertex_positions']))
    # print(params['OBJMesh.vertex_positions'])
    print('Iteration %03i: error=%g' % (it, err_ref[0]), end='\r')