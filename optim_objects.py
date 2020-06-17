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
from mitsuba.python.autodiff import render, write_bitmap, SGD, Adam, MyOpt

# This test optimizes a colorful texture from a reference image.
# ref_model
ptn = 4
ptn_ref = 10

obj_name = ["Sphere_2197", "Sphere_100", "Sphere_125", "Sphere_2197_big", "face_ver2"]
obj_ref_name = ["Ear", "core", "Gun", "s08", "n25", "a03", "SweepProfile", "kunekune", "ExCylinder", "Cylinder", "face"]
obj_path = "/home/udemegane/3DCG/Objects/3D objects/obj/" + obj_name[ptn] + ".OBJ"
path = "/home/udemegane/Documents/LabProjects/output/optim_" + obj_ref_name[ptn_ref] + "/"

if not os.path.isdir(path):
    os.makedirs(path)
for ite in range(1):
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




                            </scene>
                        """

    """<bsdf type="diffuse" id="box">
                                    <spectrum name="reflectance" value="400:0.343, 404:0.445, 408:0.551, 412:0.624, 416:0.665, 420:0.687, 424:0.708, 428:0.723, 432:0.715, 436:0.71, 440:0.745, 444:0.758, 448:0.739, 452:0.767, 456:0.777, 460:0.765, 464:0.751, 468:0.745, 472:0.748, 476:0.729, 480:0.745, 484:0.757, 488:0.753, 492:0.75, 496:0.746, 500:0.747, 504:0.735, 508:0.732, 512:0.739, 516:0.734, 520:0.725, 524:0.721, 528:0.733, 532:0.725, 536:0.732, 540:0.743, 544:0.744, 548:0.748, 552:0.728, 556:0.716, 560:0.733, 564:0.726, 568:0.713, 572:0.74, 576:0.754, 580:0.764, 584:0.752, 588:0.736, 592:0.734, 596:0.741, 600:0.74, 604:0.732, 608:0.745, 612:0.755, 616:0.751, 620:0.744, 624:0.731, 628:0.733, 632:0.744, 636:0.731, 640:0.712, 644:0.708, 648:0.729, 652:0.73, 656:0.727, 660:0.707, 664:0.703, 668:0.729, 672:0.75, 676:0.76, 680:0.751, 684:0.739, 688:0.724, 692:0.73, 696:0.74, 700:0.737"/>
                                </bsdf>
                                <bsdf type="diffuse" id="white">
                                    <spectrum name="reflectance" value="400:0.343, 404:0.445, 408:0.551, 412:0.624, 416:0.665, 420:0.687, 424:0.708, 428:0.723, 432:0.715, 436:0.71, 440:0.745, 444:0.758, 448:0.739, 452:0.767, 456:0.777, 460:0.765, 464:0.751, 468:0.745, 472:0.748, 476:0.729, 480:0.745, 484:0.757, 488:0.753, 492:0.75, 496:0.746, 500:0.747, 504:0.735, 508:0.732, 512:0.739, 516:0.734, 520:0.725, 524:0.721, 528:0.733, 532:0.725, 536:0.732, 540:0.743, 544:0.744, 548:0.748, 552:0.728, 556:0.716, 560:0.733, 564:0.726, 568:0.713, 572:0.74, 576:0.754, 580:0.764, 584:0.752, 588:0.736, 592:0.734, 596:0.741, 600:0.74, 604:0.732, 608:0.745, 612:0.755, 616:0.751, 620:0.744, 624:0.731, 628:0.733, 632:0.744, 636:0.731, 640:0.712, 644:0.708, 648:0.729, 652:0.73, 656:0.727, 660:0.707, 664:0.703, 668:0.729, 672:0.75, 676:0.76, 680:0.751, 684:0.739, 688:0.724, 692:0.73, 696:0.74, 700:0.737"/>
                                </bsdf>
                                <bsdf type="diffuse" id="red">
                                    <spectrum name="reflectance" value="400:0.04, 404:0.046, 408:0.048, 412:0.053, 416:0.049, 420:0.05, 424:0.053, 428:0.055, 432:0.057, 436:0.056, 440:0.059, 444:0.057, 448:0.061, 452:0.061, 456:0.06, 460:0.062, 464:0.062, 468:0.062, 472:0.061, 476:0.062, 480:0.06, 484:0.059, 488:0.057, 492:0.058, 496:0.058, 500:0.058, 504:0.056, 508:0.055, 512:0.056, 516:0.059, 520:0.057, 524:0.055, 528:0.059, 532:0.059, 536:0.058, 540:0.059, 544:0.061, 548:0.061, 552:0.063, 556:0.063, 560:0.067, 564:0.068, 568:0.072, 572:0.08, 576:0.09, 580:0.099, 584:0.124, 588:0.154, 592:0.192, 596:0.255, 600:0.287, 604:0.349, 608:0.402, 612:0.443, 616:0.487, 620:0.513, 624:0.558, 628:0.584, 632:0.62, 636:0.606, 640:0.609, 644:0.651, 648:0.612, 652:0.61, 656:0.65, 660:0.638, 664:0.627, 668:0.62, 672:0.63, 676:0.628, 680:0.642, 684:0.639, 688:0.657, 692:0.639, 696:0.635, 700:0.642"/>
                                </bsdf>
                                <bsdf type="diffuse" id="green">
                                    <spectrum name="reflectance" value="400:0.092, 404:0.096, 408:0.098, 412:0.097, 416:0.098, 420:0.095, 424:0.095, 428:0.097, 432:0.095, 436:0.094, 440:0.097, 444:0.098, 448:0.096, 452:0.101, 456:0.103, 460:0.104, 464:0.107, 468:0.109, 472:0.112, 476:0.115, 480:0.125, 484:0.14, 488:0.16, 492:0.187, 496:0.229, 500:0.285, 504:0.343, 508:0.39, 512:0.435, 516:0.464, 520:0.472, 524:0.476, 528:0.481, 532:0.462, 536:0.447, 540:0.441, 544:0.426, 548:0.406, 552:0.373, 556:0.347, 560:0.337, 564:0.314, 568:0.285, 572:0.277, 576:0.266, 580:0.25, 584:0.23, 588:0.207, 592:0.186, 596:0.171, 600:0.16, 604:0.148, 608:0.141, 612:0.136, 616:0.13, 620:0.126, 624:0.123, 628:0.121, 632:0.122, 636:0.119, 640:0.114, 644:0.115, 648:0.117, 652:0.117, 656:0.118, 660:0.12, 664:0.122, 668:0.128, 672:0.132, 676:0.139, 680:0.144, 684:0.146, 688:0.15, 692:0.152, 696:0.157, 700:0.159"/>
                                </bsdf>
                                <bsdf type="diffuse" id="light">
                                    <spectrum name="reflectance" value="400:0.78, 500:0.78, 600:0.78, 700:0.78"/>
                                </bsdf>
                                <shape type="obj">
                                    <string name="filename" value="10_inverse_rendering/cbox/meshes/cbox_luminaire.obj"/>
                                    <transform name="to_world">
                                        <translate x="0" y="-0.5" z="0"/>
                                    </transform>
                                    <ref id="light"/>
                                    <emitter type="area">
                                        <spectrum name="radiance" value="400:0, 500:8, 600:15.6, 700:18.4"/>
                                    </emitter>
                                </shape>
                                <shape type="obj">
                                    <string name="filename" value="10_inverse_rendering/cbox/meshes/cbox_luminaire.obj"/>
                                    <transform name="to_world">
                                        <translate x="0" y="-0.5" z="0"/>
                                    </transform>
                                    <ref id="light"/>
                                    <emitter type="area">
                                        <spectrum name="radiance" value="400:0, 500:8, 600:15.6, 700:18.4"/>
                                    </emitter>
                                </shape>
                                <shape type="obj">
                                    <string name="filename" value="10_inverse_rendering/cbox/meshes/cbox_floor.obj"/>
                                    <transform name="to_world">
                                        <translate x="0" y="-0.5" z="0"/>
                                    </transform>
                                    <ref id="white"/>
                                </shape>
                                <shape type="obj">
                                    <string name="filename" value="10_inverse_rendering/cbox/meshes/cbox_ceiling.obj"/>
                                    <ref id="white"/>
                                </shape>
                                <shape type="obj">
                                    <string name="filename" value="10_inverse_rendering/cbox/meshes/cbox_back.obj"/>
                                    <ref id="white"/>
                                </shape>
                                <shape type="obj">
                                    <string name="filename" value="10_inverse_rendering/cbox/meshes/cbox_greenwall.obj"/>
                                    <ref id="green"/>
                                </shape>
                                <shape type="obj">
                                    <string name="filename" value="10_inverse_rendering/cbox/meshes/cbox_redwall.obj"/>
                                    <ref id="red"/>
                                </shape>"""


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
    lr = [1, 5, 10, 16, 15, 25, 40]
    momentum = [0.01, 0.05, 0.1, 0.3, 0.25, 0.5, 0.75]
    n = 3
    m = 3
    str_opt_info = "lr,mom_" + str(lr[n]) + "," + str(momentum[m])
    finaloutputpath = path + obj_name[ptn] + "_" + str_opt_info + "/"
    if not os.path.isdir(finaloutputpath):
        os.makedirs(finaloutputpath)

    opt1 = Adam(properties, lr=.05)
    opt2 = SGD(properties, lr=lr[n - 1], momentum=momentum[m - 1])
    opt3 = MyOpt(properties, lr=lr[n - 1])
    opt = opt3
    l = 0.000000001

    for i in range(100):
        image = render(scene, spp=2)

        image_np = image.numpy().reshape(fsize[1], fsize[0], 3)
        output_file = finaloutputpath + 'out_%03i.exr' % i
        write_bitmap(finaloutputpath + 'out_%03i.png' % i, image, fsize)
        print("Writing image %s" % (output_file))
        Bitmap(image_np).write(output_file)

        # Objective function
        loss = ((ek.hsum(ek.sqr(image - image_ref))) + ek.hsum(l * ek.sqr((image) / (image_ref + 0.00001)))) / (
                    fsize[1] * fsize[0] * 3)
        print("Iteration %i: loss=%f" % (i, loss[0]))
        # print(ek.hsum(ek.hsum(ek.sqr(image - image_ref))))
        # print(ek.hsum(l * ek.sqr(image/image_ref)))
        # print((image+0.001)/(image_ref+0.001))
        csv_file = finaloutputpath + obj_name[ptn] + "_to_" + obj_ref_name[ptn_ref] + str(ite) + ".csv"
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
        plt.plot(df.index, dat, label=i)

        # 書くデータの描画が終わったら、タイトルと凡例を追加
    plt.title(finaloutputpath + obj_name[ptn] + "_to_" + obj_ref_name[ptn_ref])
    plt.legend()

    # pngファイルを出力
    plt.savefig(finaloutputpath + obj_name[ptn] + "_to_" + obj_ref_name[ptn_ref] + str(ite) + ".png" )
    plt.close()

