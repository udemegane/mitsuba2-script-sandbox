import os
import time
import enoki as ek

import mitsuba

mitsuba.set_variant('gpu_autodiff_rgb')

from mitsuba.core import Thread, xml, UInt32, Float, Vector2f, Vector3f, Transform4f, ScalarTransform4f
from mitsuba.render import SurfaceInteraction3f
from mitsuba.python.util import traverse
from mitsuba.python.autodiff import render, write_bitmap, Adam

def loop(args, scene, optimizer, spp, unbiased, pre_render_callback, iter, crop_size, image_ref):
    opt = optimizer
    for it in range(iter):
        # Perform a differentiable rendering of the scene
        image = render(scene,
                       optimizer=opt,
                       spp=spp,
                       unbiased=unbiased,
                       pre_render_callback=pre_render_callback)

        write_bitmap(args.out + 'out_%03i.png' % it, image, crop_size)

        # Objective: MSE between 'image' and 'image_ref'
        ob_val = ek.hsum(ek.sqr(image - image_ref)) / len(image)

        # Back-propagate errors to input parameters
        ek.backward(ob_val)

        # Optimizer: take a gradient step -> update displacement map
        opt.step()

        # Compare iterate against ground-truth value
        err_ref = ek.hsum(ek.sqr(disp_tex_data_ref - disp_tex.eval_1(mesh_si)))
        print('Iteration %03i: error=%g' % (it, err_ref[0]), end='\r')
