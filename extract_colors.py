import os
import sys
import argparse
import pickle
import glob
import colorgram

import numpy as np
import pandas as pd
import seaborn as sns

from tqdm import tqdm
from joblib import Parallel, delayed

sys.path.append('../')
from utils.misc import check_exists_makedirs, batchify_iterable, rgb2hex, extract_colors

def build_opt(parser):
    
    parser.add_argument('-input_dir', type=str, required=True,
                        help='Root directory of `png` folder generated from `generate_chart.ipynb`. It will output `colors/` in this directory too.')
    parser.add_argument('-core', type=str, default='extcolors',
                        help='Color extraction library to use. Options: ["colorgram", "extcolors"]')
    parser.add_argument('-n_colors', type=int, default=10,       
                        help='Number of colors to extract from a chart image.')
    parser.add_argument('-n_jobs', type=int, default=20,
                        help='Number of cpus to perform multi-process.')
    parser.add_argument('-batch_size', type=int, default=10000,
                        help='Batch size to process for a single cpu during multi-process.')
    
    opt = parser.parse_args()
    return opt
    
def process(img_path, output_dir, n_colors, core):
    output_name = os.path.splitext(os.path.basename(img_path))[0]
    output_path = os.path.join(output_dir, f'{output_name}.pkl')
    obj = extract_colors(img_path, n_colors, core=core)

    with open(output_path, 'wb') as file:
        pickle.dump(obj, file)

def main(opt):
    # Load images (generated from `generate_chart.ipynb`)
    img_dir = os.path.join(opt.input_dir, 'png')
    img_paths = glob.glob(os.path.join(img_dir, '*.png'))
    print(f'Number of image: {len(img_paths)}')

    # Make output colors directory.
    color_dir = os.path.join(opt.input_dir, 'colors')
    check_exists_makedirs(color_dir, warning=False)
    print(f'Make directory: "{color_dir}"')
    
    # Start multi-process generating.
    with Parallel(n_jobs=opt.n_jobs) as parallel:

        num_iteration = len(img_paths) // opt.batch_size + 1
        pbar = tqdm(batchify_iterable(img_paths, batch_size=opt.batch_size))

        for batch_id, batch in enumerate(pbar):
            pbar.set_description('Extracting colors batch: ({}/{})'.format(batch_id+1, num_iteration))
            parallel(delayed(process)(img_path, color_dir, opt.n_colors, opt.core) for img_path in batch)
    
if __name__ == '__main__':
    """ Usage example:
        $ python extract_colors.py -input_dir ../data/ijcai19_v0/train/vbar/
        
        Note:
        - There's a `png` folder in "../data/ijcai19_v0/train/vbar/" which contains chart images.
        - It will output `colors` folder in "../data/ijcai19_v0/train/vbar/" which contains extracted colors.
    """
    
    parser = argparse.ArgumentParser(
        description='extract_colors.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    opt = build_opt(parser)
    
    main(opt)

    
    



