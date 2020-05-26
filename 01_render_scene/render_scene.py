import os
import numpy as np
import mitsuba

# Set the desired mitsuba variant
mitsuba.set_variant('scalar_rgb')

from mitsuba.core import Bitmap, Struct, Thread
from mitsuba.core.xml import load_file

# Absolute or relative path to the XML file
filename = '/home/udemegane/mitsuba2/docs/testscripts/10_inverse_rendering/cbox/cbox.xml'

# Add the scene directory to the FileResolver's search path
Thread.thread().file_resolver().append(os.path.dirname(filename))

# Load the actual scene
scene = load_file(filename)

# Call the scene's integrator to render the loaded scene
scene.integrator().render(scene, scene.sensors()[0])

# After rendering, the rendered data is stored in the film
film = scene.sensors()[0].film()
print(type(film))
# Write out rendering as high dynamic range OpenEXR file
film.set_destination_file('/home/udemegane/mitsuba2/docs/testscripts/01_render_scene/out.exr')
film.develop()
print(type(film))

# Write out a tonemapped JPG of the same renderingane
bmp = film.bitmap(raw=True)
bmp.convert(Bitmap.PixelFormat.RGB, Struct.Type.UInt8, srgb_gamma=True).write('/home/udemegane/mitsuba2/docs/testscripts/01_render_scene/out.png')

# Get linear pixel values as a numpy array for further processing
bmp_linear_rgb = bmp.convert(Bitmap.PixelFormat.RGB, Struct.Type.Float32, srgb_gamma=False)
image_np = np.array(bmp_linear_rgb)
print(image_np.shape)
