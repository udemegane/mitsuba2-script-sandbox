import numpy as np
import os
import csv
import mitsuba
import enoki as ek

from matplotlib import pyplot as plt
import pandas as pd

mts_variant = 'rgb'
mitsuba.set_variant('gpu_autodiff_' + mts_variant)

from mitsuba.core import Transform4f, Bitmap, Float, Vector3f, Struct
from mitsuba.core.xml import load_string
from mitsuba.python.util import traverse
from mitsuba.python.autodiff import render, write_bitmap, SGD, Adam

# This test optimizes a colorful texture from a reference image.
for ptn in range(3):
    for ptn_ref in range(10):
        obj_name = ["Sphere_125", "Sphere_100", "Sphere_8000"]
        obj_ref_name = ["kunekune", "ExCylinder", "Cylinder", "Ear", "core", "Gun", "s08", "n25", "a03", "SweepProfile"]
        obj_path = "/home/udemegane/3DCG/Objects/3D objects/obj/" + obj_name[ptn] + ".OBJ"
        path = "output/optim_" + obj_ref_name[ptn_ref] + "/"

        if not os.path.isdir(path):
            os.makedirs(path)

        xml_string = """
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

                        <shape type="obj" id="planemesh">
                            <string name="filename" value="data/meshes/xy_plane.obj"/>

                            <bsdf type="diffuse" id="planemat">
                            </bsdf>

                            <transform name="to_world">
                                <translate z="-1"/>
                                <scale value="2.0"/>
                            </transform>    
                        </shape>
                    </scene>
                """

        for n in range(6):
            for m in range(6):
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

                scene = make_scene(path_str, 8);
                fsize = scene.sensors()[0].film().size()

                # Road the ref image
                img_ref_name = obj_ref_name[ptn_ref] + "_ref" + ".exr"
                bitmap_tmp = Bitmap(path + img_ref_name, Bitmap.FileFormat.OpenEXR).convert(Bitmap.PixelFormat.RGB,
                                                                                            Struct.Type.Float32,
                                                                                            srgb_gamma=False)
                image_ref = np.array(bitmap_tmp).flatten()
                print("Writing " + path + img_ref_name)

                # Define the differentiable scene for the optimization
                del scene
                scene = make_scene(path_reparam_str, 7);

                properties = traverse(scene)
                print("list of properties:")
                print(properties)

                key = "object.vertex_positions"
                properties.keep([key])
                print("selected property value")
                print(properties[key])

                # Instantiate an optimizer
                lr = [1, 5, 10, 15, 25, 40]
                momentum = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75]
                str_opt_info = "lr,mom_" + str(lr[n]) + "," + str(momentum[m])
                finaloutputpath = path + obj_name[ptn] + "_" + str_opt_info + "/"
                if not os.path.isdir(finaloutputpath):
                    os.makedirs(finaloutputpath)

                opt1 = Adam(properties, lr=.05)
                opt2 = SGD(properties, lr=lr[n - 1], momentum=momentum[m - 1])
                opt = opt2

                for i in range(100):
                    image = render(scene, spp=2)

                    image_np = image.numpy().reshape(fsize[1], fsize[0], 3)
                    output_file = finaloutputpath + 'out_%03i.exr' % i
                    write_bitmap(finaloutputpath + 'out_%03i.png' % i, image, fsize)
                    print("Writing image %s" % (output_file))
                    Bitmap(image_np).write(output_file)

                    # Objective function
                    loss = ek.hsum(ek.hsum(ek.sqr(image - image_ref))) / (fsize[1] * fsize[0] * 3)
                    print("Iteration %i: loss=%f" % (i, loss[0]))

                    csv_file = finaloutputpath + obj_name[ptn] + "_to_" + obj_ref_name[ptn_ref] + ".csv"
                    try:
                        with open(csv_file, mode='x') as f:
                            csv.writer(f).writerow([i, loss[0]])
                    except FileExistsError:
                        with open(csv_file, 'a') as f:
                            csv.writer(f).writerow([i, loss[0]])

                    ek.backward(loss)
                    opt.step()
                ##make graph
                df = pd.read_csv(csv_file, index_col=0)
                for i, dat in df.iteritems():
                    # x軸がindex(時間)、y軸が書くデータの値
                    plt.plot(dat, df.index, label=i)

                    # 書くデータの描画が終わったら、タイトルと凡例を追加
                plt.title(finaloutputpath + obj_name[ptn] + "_to_" + obj_ref_name[ptn_ref])
                plt.legend()

                # pngファイルを出力
                plt.savefig(finaloutputpath + obj_name[ptn] + "_to_" + obj_ref_name[ptn_ref] + ".png")
                plt.close()
