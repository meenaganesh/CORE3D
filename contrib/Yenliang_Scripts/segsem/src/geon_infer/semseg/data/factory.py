import argparse
from os.path import join
import json
#import sys
#sys.path.append('./')

from geon_infer.semseg.data.potsdam import (POTSDAM, PotsdamImageFileGenerator, PotsdamNumpyFileGenerator)
from geon_infer.semseg.options import SemsegOptions 

from geon_infer.semseg.data.generators import NUMPY, IMAGE, TRAIN, VALIDATION
from geon_infer.semseg.data.utils import plot_sample
from geon_infer.common.utils import _makedirs

import pdb

def get_data_generator(options): 
    if options.dataset_name == POTSDAM:
        if options.generator_name == NUMPY:
            return PotsdamNumpyFileGenerator(
                options.dataset_path, options.active_input_inds,
                options.train_ratio)
        
        elif options.generator_name == IMAGE:
            return PotsdamImageFileGenerator(
                options.dataset_path, options.active_input_inds,
                options.train_ratio)
        else:
            raise ValueError('{} is not a valid generator'.format(
                options.generator_name))
    else:
      raise ValueError('not supported dataset name')


def plot_generator(dataset_name, generator_name, split):
    nb_batches = 2
    batch_size = 4

    class Options():
        def __init__(self):
            self.dataset_name = dataset_name
            self.generator_name = generator_name
            self.active_input_inds = [0, 1, 2]
            if dataset_name == POTSDAM:
                self.active_input_inds = [0, 1, 2]
            self.train_ratio = 0.8
            self.cross_validation = None

    options = Options()
    generator = get_data_generator(options, datasets_path)

    viz_path = join(
        results_path, 'gen_samples', dataset_name, generator_name, split)
    _makedirs(viz_path)

    gen = generator.make_split_generator(
        TRAIN, target_size=(400, 400), batch_size=batch_size, shuffle=True,
        augment=True, normalize=True, eval_mode=True) 
    
    for batch_ind in range(nb_batches):
        _, batch_y, all_batch_x, _, _ = next(gen)
        for sample_ind in range(batch_size):
            file_path = join(
                viz_path, '{}_{}.pdf'.format(batch_ind, sample_ind))
            plot_sample(
                file_path, all_batch_x[sample_ind, :, :, :],
                batch_y[sample_ind, :, :, :], generator)


def preprocess(file_path):
    with open(file_path) as options_file:
      options_dict = json.load(options_file)
      options = SemsegOptions(options_dict)
      PotsdamImageFileGenerator.preprocess(options.dataset_path)
      PotsdamNumpyFileGenerator.preprocess(options.dataset_path)

def plot_generators(): 
    plot_generator(POTSDAM, NUMPY, TRAIN)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('file_path', nargs='?')
    parser.add_argument('--preprocess',action='store_true') 
    parser.add_argument('--plot',action='store_true')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    if args.preprocess:
        preprocess(args.file_path)

    if args.plot:
        plot_generators()
