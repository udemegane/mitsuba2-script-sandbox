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

def repeater(iterations, scene, pre_render_callback, optimizer):
    for it in range(iterations):
        # Perform a differentiable rendering of the scene
        image = render(scene,
                       optimizer=optimizer,
                       spp=2,
                       unbiased=True,
                       pre_render_callback=pre_render_callback)

        write_bitmap(output_path + 'out_%03i.exr' % it, image, crop_size)

        # Objective: MSE between 'image' and 'image_ref'
        ob_val = ek.hsum(ek.sqr(image - image_ref)) / len(image)

        # Back-propagate errors to input parameters
        ek.backward(ob_val)

        # Optimizer: take a gradient step -> update displacement map
        opt.step()

        print('Iteration %03i: error=%g' % (it, ob_val[0]), end='\r')