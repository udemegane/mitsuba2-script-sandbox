from pathlib import Path
import mitsuba
import enoki as ek
mitsuba.set_variant('gpu_autodiff_rgb')

from mitsuba.core import Bitmap, Float, Vector3f, Struct, UInt32
from mitsuba.core.xml import load_string
from mitsuba.python.autodiff import render, write_bitmap, SGD
from mitsuba.python.util import traverse

# Convert flat array into a vector of arrays (will be included in next enoki release)
def ravel(buf, dim = 3):
    idx = dim * UInt32.arange(ek.slices(buf) // dim)
    return Vector3f(ek.gather(buf, idx), ek.gather(buf, idx + 1), ek.gather(buf, idx + 2))

# Return contiguous flattened array (will be included in next enoki release)
def unravel(source, target, dim = 3):
    idx = UInt32.arange(ek.slices(source))
    for i in range(dim):
        ek.scatter(target, source[i], dim * idx + i)

def make_scene(integrator, mesh, spp, w=256, h=256):
    return load_string(
        """
        <?xml version="1.0" encoding="utf-8"?>
        <scene version="0.4.0">
        	{integrator}

	        <sensor type="perspective">
		        <transform name="toWorld">
			        <lookAt origin="0, 0, 3" target="0, 0, 0" up="0, 1, 0"/>
		        </transform>
		        <float name="fov" value="45"/>

                <sampler type="independent">
                    <integer name="sampleCount" value="{spp}"/>
                </sampler>

                <film type="hdrfilm">
                    <integer name="width" value="{w}"/>
                    <integer name="height" value="{h}"/>

                    <rfilter type="box"/>
                </film>
	        </sensor>

            <shape type="obj" id="sheet">
                <string name="filename" value="{mesh}.obj"/>
                <bsdf type="roughplastic">
                    <float name="alpha" value="0.01"/>
                </bsdf>
            </shape>

            <shape type="obj" id="light">
                <string name="filename" value="grid.obj"/>
                <transform name="toWorld">
                    <rotate y="1" angle="180"/>
                    <translate x="0" y="0" z="15"/>
                </transform>
                <emitter type="smootharea">
                    <spectrum name="radiance" value="100"/>
                </emitter>
            </shape>
        </scene>
        """.format(integrator=integrator, mesh=mesh, spp=spp, w=w, h=h)
    )

path_str =  """
<integrator type="path">
    <integer name="max_depth" value="2"/>
</integrator>"""

path_reparam_str =  """
<integrator type="pathreparam">
    <integer name="max_depth" value="2"/>
    <boolean name="use_variance_reduction" value="true"/>
    <boolean name="use_convolution" value="true"/>
    <boolean name="disable_gradient_diffuse" value="false"/>
</integrator>"""

width   = 150
height  = 150
spp_ref = 64
spp_opt = 4
out_dir = Path("/home/udemegane/Documents/LabProjects/output/plane")
out_dir.mkdir(parents=True, exist_ok=True)

# Generate the reference scene and image
scene = make_scene(path_str, "grid_ref", spp_ref, width, height)
image_ref = render(scene)

write_bitmap(str(out_dir / "out_ref.png"), image_ref, (width, height))
print("Writing " + "out_ref.png")

# Generate the scene to use for optimization
del scene
scene = make_scene(path_reparam_str, "grid", spp_opt, width, height)

vertex_pos_key = 'sheet.vertex_positions_buf'

params = traverse(scene)
params.keep([vertex_pos_key])
print("Parameter map after filtering: ", params)
#
vertex_positions_buf = params[vertex_pos_key]
vertex_positions = ravel(vertex_positions_buf)
vertex_count = ek.slices(vertex_positions)

# Initialize the estimated displacement vector
displacements = ek.full(Float, 0.0, vertex_count)

params_opt = {"displacements": displacements}

# Instantiate an optimizer
opt = SGD(params_opt, lr=1.0, momentum=0.9)

for i in range(100):
    # Update the scene
    unravel(vertex_positions + Vector3f(0, 0, 1) * params_opt['displacements'], params[vertex_pos_key])
    params.set_dirty(vertex_pos_key)
    params.update()

    image = render(scene)
    
    if ek.any(ek.any(ek.isnan(params[vertex_pos_key]))):
        print("[WARNING] NaNs in the vertex positions.")

    if ek.any(ek.isnan(image)):
        print("[WARNING] NaNs in the image.")

    # Write a gamma encoded PNG
    image_np = image.numpy().reshape(height, width, 3)
    output_file = str(out_dir / 'out_{:03d}.png'.format(i))
    print("Writing image %s" % (output_file))
    Bitmap(image_np).convert(pixel_format=Bitmap.PixelFormat.RGB, component_format=Struct.Type.UInt8, srgb_gamma=True).write(output_file)

    # Objective function
    loss = ek.hsum(ek.hsum(ek.sqr(image - image_ref))) / (height*width*3)
    print("Iteration %i: loss=%f" % (i, loss[0]))

    if(loss[0] != loss[0]):
        print("[WARNING] Skipping current iteration due to NaN loss.")
        continue

    ek.backward(loss)

    if ek.any(ek.isnan(ek.gradient(params_opt['displacements']))):
            print("[WARNING] NaNs in the displacement gradients. ({iteration:d})".format(iteration=i))
            exit(-1)

    opt.step()

    if ek.any(ek.isnan(params_opt['displacements'])):
        print("[WARNING] NaNs in the vertex displacements. ({iteration:d})".format(iteration=i))
        exit(-1)