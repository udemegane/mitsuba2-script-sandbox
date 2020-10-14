import os
import numpy as np
import enoki as ek

import mitsuba
mitsuba.set_variant('gpu_autodiff_rgb')

from mitsuba.core import Thread, xml, UInt32, Struct, Float, Vector2f, Vector3f, Bitmap, Transform4f, ScalarTransform4f
from mitsuba.render import SurfaceInteraction3f
from mitsuba.python.util import traverse
from mitsuba.python.autodiff import render, write_bitmap, Adam
from collections import namedtuple

def ravel(buf, dim = 3):
    idx = dim * UInt32.arange(ek.slices(buf) // dim)
    if dim == 2:
        return Vector2f(ek.gather(buf, idx), ek.gather(buf, idx + 1))
    elif dim == 3:
        return Vector3f(ek.gather(buf, idx), ek.gather(buf, idx + 1), ek.gather(buf, idx + 2))

# Return contiguous flattened array (will be included in next enoki release)
def unravel(source, target, dim = 3):
    idx = UInt32.arange(ek.slices(source))
    for i in range(dim):
        ek.scatter(target, source[i], dim * idx + i)

def initialize(scene_name, img_ref_name, file_name):
    data_path = '../data/'
    output_path = data_path + 'output/' + file_name + '/'
    img_ref_path = data_path + 'refimages/' + img_ref_name
    scene_folder = data_path + 'xml/' + scene_name
    scene_folder = '../../../../resources/data/docs/examples/invert_heightfield/'

    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    Thread.thread().file_resolver().append(scene_folder)
    scene = xml.load_file(scene_folder + 'scene.xml')

    # Load a reference image (no derivatives used yet)
    crop_size = scene.sensors()[0].film().crop_size()
    bitmap_tmp = Bitmap(output_path + img_ref_name, Bitmap.FileFormat.OpenEXR).convert(Bitmap.PixelFormat.RGB,
                                                                                       Struct.Type.Float32,
                                                                                       srgb_gamma=False)
    img_ref = np.array(bitmap_tmp).flatten()

    data = namedtuple('data', ['scene', 'img_ref', 'crop_size', 'output_path'])

    return data

