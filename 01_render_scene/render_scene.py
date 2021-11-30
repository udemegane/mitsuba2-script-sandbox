import os
import numpy as np
import mitsuba
import enoki as ek
# Set the desired mitsuba variant
mitsuba.set_variant('gpu_autodiff_rgb')

from mitsuba.core import Bitmap, Struct, Thread
from mitsuba.core.xml import load_file

from mitsuba.core import Float, Thread
from mitsuba.core.xml import load_file
from mitsuba.python.util import traverse
from mitsuba.python.autodiff import render, write_bitmap, Adam
import time
# Absolute or relative path to the XML file
mitsuba_dir = '/home/udemegane/workspace/mitsuba2/'
myscript_dir = '/home/udemegane/workspace/mitsuba2-script-sandbox/'
filename = myscript_dir + 'data/xml/bsdf_blendbsdf.xml'

# Add the scene directory to the FileResolver's search path
Thread.thread().file_resolver().append(os.path.dirname(filename))

# Load the actual scene
scene = load_file(filename)

# Call the scene's integrator to render the loaded scene
image_ref = render(scene, spp=20)
crop_size = scene.sensors()[0].film().crop_size()
write_bitmap('output.png', image_ref, crop_size)