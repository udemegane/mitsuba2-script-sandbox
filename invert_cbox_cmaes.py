# Simple inverse rendering example: render a cornell box reference image,
# then replace one of the scene parameters and try to recover it using
# differentiable rendering and gradient-based optimization.

import enoki as ek
import mitsuba
import numpy as np
mitsuba.set_variant('gpu_autodiff_rgb')
from cmaes import CMA
from mitsuba.core import Thread, Color3f
from mitsuba.core.xml import load_file
from mitsuba.python.util import traverse
from mitsuba.python.autodiff import render, write_bitmap, Adam
import time
import matplotlib.pyplot as plt
from copy import copy
# Load the Cornell Box
Thread.thread().file_resolver().append('cbox')
scene = load_file('cbox/cbox.xml')

# Find differentiable scene parameters
params = traverse(scene)
print(params)

# Discard all parameters except for one we want to differentiate
# params.keep(['box.eta'])
params.keep(['red.reflectance.value'])
# Print the current value and keep a backup copy
param_ref = params["red.reflectance.value"]
print(param_ref)

# Render a reference image (no derivatives used yet)
image_ref = render(scene, spp=8)
print(type(image_ref))
np_test = np.array(image_ref)
print(type(np_test))
crop_size = scene.sensors()[0].film().crop_size()
write_bitmap('out_ref.png', image_ref, crop_size)

# Change the left wall into a bright white surface
#params["box.eta"] = 2.0
params['red.reflectance.value'] = [.9, .9, .9]
params.update()

# Construct an Adam optimizer that will adjust the parameters 'params'
#opt = Adam(params, lr=0.1)
optimizer = CMA(mean=np.array((0.0, 0.0, 0.0)), sigma=0.8)
time_a = time.time()

for it in range(100):
    solutions = []
    flag = True
    for _ in range(optimizer.population_size):
        image = render(scene, spp=3)
        if flag:
            write_bitmap('out_%03i.png' % it, image, crop_size)
            flag = False
        x = optimizer.ask()
        value = (np.square(np.array(image) -
                 np.array(image_ref))).mean(axis=None)  # / \
        #    np.size(np.array(image))
        print(value)
        # ek.hsum(ek.sqr(image - image_ref)) / len(image)
        solutions.append((x, value))
        params['red.reflectance.value'] = [x[0], x[1], x[2]]
        params.update()
    # print(solutions)
    optimizer.tell(solutions)
"""
eta_array = []
x = []
i = 0
iterations = 100
for it in range(iterations):
    # Perform a differentiable rendering of the scene
    image = render(scene, optimizer=opt, unbiased=True, spp=1)
    write_bitmap('out_%03i.png' % it, image, crop_size)

    # Objective: MSE between 'image' and 'image_ref'
    ob_val = ek.hsum(ek.sqr(image - image_ref)) / len(image)

    # Back-propagate errors to input parameters
    ek.backward(ob_val)

    # Optimizer: take a gradient step
    opt.step()

    # Compare iterate against ground-truth value
    err_ref = ek.hsum(ek.sqr(param_ref - params["box.eta"]))
    print('Iteration %08i: error=%g' % (it, err_ref[0]), end='\r')
    print(params["box.eta"])
    tmp = params["box.eta"]
    eta_array.append(tmp)
    x.append(i)
    i = i + 1
print(eta_array)
plt.plot(x, eta_array)
plt.savefig("graph.png")
time_b = time.time()

print()
print('%f ms per iteration' % (((time_b - time_a) * 1000) / iterations))
"""
