import argparse
import os

import mitsuba
mitsuba.set_variant('gpu_autodiff_rgb')

#import mitsuba c++ modules
from mitsuba.core import xml, Thread

#import miitsuba python modules
from mitsuba.python.autodiff import render, write_bitmap

def run(scene_path, out_dir, spp, pp_name):
    Thread.thread().file_resolver().append(scene_path)
    scene = xml.load_file(scene_path)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    image_raw = render(scene, spp)
    crop_size = scene.sensors()[0].film().crop_size()
    write_bitmap(out_dir + scene_path.split('.')[0] + '.png', image_raw, crop_size)
    print('Write ' + out_dir + scene_path.split('.')[0] + '.png')

def main():
    parser = argparse.ArgumentParser(
        description='Render Scene by Nvidia GPU.',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--scene', default='../data/scene/cbox.xml')
    parser.add_argument('--out', default='../data/output/cbox/')
    parser.add_argument('--spp', default='16')
    parser.add_argument('--post_process', default='None')
    args = parser.parse_args()

    run(**vars(args))

if __name__ == '__main__':
    main()
