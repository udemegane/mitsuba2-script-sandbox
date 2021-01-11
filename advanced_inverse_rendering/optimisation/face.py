import time

import enoki as ek
import mitsuba

mitsuba.set_variant('gpu_autodiff_rgb')

from mitsuba.core import Thread, xml, UInt32, Float, Vector2f, Vector3f, Transform4f, ScalarTransform4f
from mitsuba.render import SurfaceInteraction3f
from mitsuba.python.util import traverse
from mitsuba.python.autodiff import render, write_bitmap, Adam

from mylib.util import ravel, unravel


def optimisation(args, scene, params):
    positions_initial = ravel(params['grid_mesh.vertex_positions_buf'])
    normals_initial = ravel(params['grid_mesh.vertex_normals_buf'])
    vertex_count = ek.slices(positions_initial)

    # Create a texture with the reference displacement map
    disp_tex = xml.load_dict({
        "type": "bitmap",
        "filename": "mitsuba_coin.jpg",
        "to_uv": ScalarTransform4f.scale([1, -1, 1])  # texture is upside-down
    }).expand()[0]

    # Create a fake surface interaction with an entry per vertex on the mesh
    mesh_si = SurfaceInteraction3f.zero(vertex_count)
    mesh_si.uv = ravel(params['grid_mesh.vertex_texcoords_buf'], dim=2)

    # Evaluate the displacement map for the entire mesh
    disp_tex_data_ref = disp_tex.eval_1(mesh_si)