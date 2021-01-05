import argparse
import os
import yaml
import mitsuba

mitsuba.set_variant('gpu_autodiff_rgb')

# import mitsuba c++ modules
from mitsuba.core import xml, Thread

# import mitsuba python modules
from mitsuba.python.autodiff import render, write_bitmap

from optimisation import heightfield
from mylib.util import EasyDict


def run(args):
    Thread.thread().file_resolver().append(args.scene_dir)
    scene = xml.load_file(args.scene_dir + 'scene.xml')
    if not os.path.exists(args.out):
        os.makedirs(args.out)

    # image_raw = render(scene, args.spp)
    # crop_size = scene.sensors()[0].film().crop_size()
    # write_bitmap(args.out + args.scene.split('.')[0] + '.png', image_raw, crop_size)
    # print('Write ' + args.out + args.scene.split('.')[0] + '.png')
    if args.script + '.py' in os.listdir('./optimisation'):
        eval(args.script + '.optimisation')(args, scene)
    else:
        print('Error: Invalid scripts "' + args.script + '"')


def main():
    parser = argparse.ArgumentParser(
        description='Render Scene by Nvidia GPU.',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--scene_dir', default=None)
    parser.add_argument('--ref_image', default=None)
    parser.add_argument('--out', default=None)
    parser.add_argument('--spp', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--iter', type=int, default=None)
    parser.add_argument('--configfile', default='./config/heightfield.yml')
    parser.add_argument('--script', default=None)
    args = parser.parse_args()

    with open(args.configfile, 'r') as file:
        tmp = yaml.load(file)
        config = EasyDict(tmp)
    config.spp = int(config.spp)
    config.iter = int(config.iter)
    config.lr = float(config.lr)



    run(config)

if __name__ == '__main__':
    main()
