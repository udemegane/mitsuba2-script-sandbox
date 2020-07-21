import numpy as np
import os
import mitsuba
import enoki as ek

mts_variant = 'rgb'
mitsuba.set_variant('gpu_autodiff_' + mts_variant)

from mitsuba.core import Transform4f, Bitmap, Float, Vector3f, Struct
from mitsuba.core.xml import load_string
from mitsuba.python.autodiff import render, write_bitmap

roop = 12
for i in range(roop):
    #obj_name = "s%02i" % (i+1)
    obj_name = ["kunekune", "ExCylinder", "Cylinder", "Ear", "core", "Gun", "s08", "n25", "a03", "SweepProfile", "face", "face_ver2"]
    obj_polyhedrons = "polyhedrons_obj/" + obj_name[i] + ".obj"
    obj_path_sub = obj_polyhedrons
    obj_path_sub = obj_name[i] + ".OBJ"
    obj_path = "/home/udemegane/3DCG/Objects/3D objects/obj/" + obj_path_sub
    path = "/home/udemegane/Documents/LabProjects/output/optim_" + obj_name[i] + "/"

    if not os.path.isdir(path):
        os.makedirs(path)

    xml_string = """x
                           <?xml version="1.0"?>
                           <scene version="2.0.0">
                               {integrator}
                               <sensor type="perspective">
                                   <string name="fov_axis" value="smaller"/>
                                   <float name="near_clip" value="0.1"/>
                                   <float name="far_clip" value="2800"/>
                                   <float name="focus_distance" value="1000"/>

                                   <transform name="to_world">
                                       <lookat origin="0, 0, 9" target="0, 0, 0" up="0, 1, 0"/>
                                   </transform>
                                   <float name="fov" value="15"/>
                                   <sampler type="independent">
                                       <integer name="sample_count" value="{spp}"/>
                                   </sampler>
                                   <film type="hdrfilm">
                                       <integer name="width" value="250"/>
                                       <integer name="height" value="250"/>
                                       <rfilter type="box" >
                                           <float name="radius" value="0.5"/>
                                       </rfilter>
                                   </film>
                               </sensor>

                               <shape type="obj" id="smooth_area_light_shape">
                                   <transform name="to_world">
                                       <rotate x="1" angle="180"/>
                                       <translate x="10.0" y="0.0" z="15.0"/>
                                   </transform>
                                   <string name="filename" value="data/meshes/xy_plane.obj"/>
                                   <emitter type="smootharea" id="smooth_area_light">
                                       <spectrum name="radiance" value="100"/>
                                   </emitter>
                                </shape>
                                    <shape type="obj" id="object">
                                    <string name="filename" value=""" + "\"" + obj_path + "\"" + """/>
                                    <bsdf type="diffuse" id="objectmat">
                                    </bsdf>
                                    <transform name="to_world">
                                    <translate z="0.0"/>
                                    </transform>    
                                </shape>




                                </scene>
                            """


    def make_scene(integrator, spp):
        return load_string(xml_string.format(integrator=integrator, spp=spp))


    # Define integrators for this test

    path_str = """<integrator type="path">
                       <integer name="max_depth" value="2"/>
                   </integrator>"""

    path_reparam_str = """<integrator type="pathreparam">
                               <integer name="max_depth" value="2"/>
                           </integrator>"""

    if not os.path.isdir(path):
        os.makedirs(path)

    scene = make_scene(path_str, 64);
    fsize = scene.sensors()[0].film().size()

    # Render the target image
    image_ref = render(scene)
    write_bitmap(path + obj_name[i] + "_ref_test2" + ".exr", image_ref, fsize)
    print("Writing " + path + obj_name[i] + "_ref_original" + ".exr")
