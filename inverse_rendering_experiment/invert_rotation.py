import os
import time
import enoki as ek

import mitsuba
mitsuba.set_variant('gpu_autodiff_rgb')

from mitsuba.core import Thread, xml, UInt32, Float, Vector2f, Vector3f, Transform4f, ScalarTransform4f
from mitsuba.render import SurfaceInteraction3f
from mitsuba.python.util import traverse
from mitsuba.python.autodiff import render, write_bitmap, Adam

from invert import ravel, unravel, initialize

if __name__ == "__main__":
    vals = initialize('invert_heightfield', 'heigthfield_ref', os.path.splitext(os.path.basename(__file__))[0])
    scene = vals[0]
    output_path = vals[1]
    #image_ref = vals[2]
    params = traverse(scene)
    print(params)

    positions_buf = params['grid_mesh.vertex_positions_buf']
    positions_initial = ravel(positions_buf)
    normals_initial = ravel(params['grid_mesh.vertex_normals_buf'])
    vertex_count = ek.slices(positions_initial)

    disp_tex = xml.load_dict({
        "type": "bitmap",
        "filename": "mitsuba_coin.jpg",
        "to_uv": ScalarTransform4f.scale([1, -1, 1])  # texture is upside-down
    }).expand()[0]

    mesh_si = SurfaceInteraction3f.zero(vertex_count)
    mesh_si.uv = ravel(params['grid_mesh.vertex_texcoords_buf'], dim=2)

    # Evaluate the displacement map for the entire mesh
    disp_tex_data_ref = disp_tex.eval_1(mesh_si)


    def apply_displacement(amplitude=0.05):
        new_positions = disp_tex.eval_1(mesh_si) * normals_initial * amplitude + positions_initial
        unravel(new_positions, params['grid_mesh.vertex_positions_buf'])
        params.set_dirty('grid_mesh.vertex_positions_buf')
        params.update()


    # Apply displacement before generating reference image
    apply_displacement()

    # Render a reference image (no derivatives used yet)
    image_ref = render(scene, spp=32)
    crop_size = scene.sensors()[0].film().crop_size()
    write_bitmap('../data/refimages/' + 'out_ref.exr', image_ref, crop_size)
    print("Write " + '../data/refimages/' + "out_ref.exr")

    # Reset texture data to a constant
    disp_tex_params = traverse(disp_tex)
    disp_tex_params.keep(['data'])
    disp_tex_params['data'] = ek.full(Float, 0.25, len(disp_tex_params['data']))
    disp_tex_params.update()

    # Construct an Adam optimizer that will adjust the texture parameters
    opt = Adam(disp_tex_params, lr=0.002)

    time_a = time.time()

    iterations = 100
    for it in range(iterations):
        # Perform a differentiable rendering of the scene
        image = render(scene,
                       optimizer=opt,
                       spp=4,
                       unbiased=True,
                       pre_render_callback=apply_displacement)

        write_bitmap(output_path + 'out_%03i.exr' % it, image, crop_size)

        # Objective: MSE between 'image' and 'image_ref'
        ob_val = ek.hsum(ek.sqr(image - image_ref)) / len(image)

        # Back-propagate errors to input parameters
        ek.backward(ob_val)

        # Optimizer: take a gradient step -> update displacement map
        opt.step()

        # Compare iterate against ground-truth value
        err_ref = ek.hsum(ek.sqr(disp_tex_data_ref - disp_tex.eval_1(mesh_si)))
        print('Iteration %03i: error=%g' % (it, err_ref[0]), end='\r')

    time_b = time.time()

    print()
    print('%f ms per iteration' % (((time_b - time_a) * 1000) / iterations))