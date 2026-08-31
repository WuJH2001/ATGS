import glob
import os
import cv2
from argparse import ArgumentParser
from arguments.atgs_cfg import cfg

def images2video(image_paths, output_path, fps):
    import imageio

    images = []
    for image_path in image_paths:
        image = imageio.imread(image_path)
        images.append(image)

    # imageio.mimwrite(os.path.join(model_path, name, "ours_{}".format(iteration), 'videos', str(render_view_id), f'{global_start_frames}_{global_end_frames}.mp4'), render_images,fps=25)
    imageio.mimwrite(output_path, images, fps=fps)

def imageSortedFn(item):
    return int(item.split('/')[-1].split('.png')[0])


import sys

if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")

    parser.add_argument("--input_dir_path", type=str, required=True)
    parser.add_argument("--fps", type=int, default=30)

    args = parser.parse_args(sys.argv[1:])

    print(args.input_dir_path)
    image_paths = glob.glob(f'{args.input_dir_path}/*.png')

    image_paths = sorted(image_paths, key=imageSortedFn)[0:]

    output_path = f'{args.input_dir_path}/0_{len(image_paths)}.mp4'

    images2video(image_paths, output_path, args.fps)