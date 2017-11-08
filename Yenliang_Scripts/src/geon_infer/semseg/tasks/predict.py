from os.path import join
from shutil import rmtree

import numpy as np

from ..data.generators import VALIDATION, TEST
from .utils import make_prediction_img, predict_x
from geon_infer.common.utils import _makedirs, save_img
import pdb


VALIDATION_PREDICT = 'validation_predict'
TEST_PREDICT = 'test_predict'

def predict(run_path, model, options, generator, split):
    dataset = generator.dataset
    
    predictions_path = join(run_path, '{}_predictions'.format(split))
    _makedirs(predictions_path)

    split_gen = generator.make_split_generator(
        split, options.model_type, target_size=None,
        batch_size=1, shuffle=False, augment=False, normalize=True,
        eval_mode=True)

    for sample_ind, (batch_x, _, _, _, _, file_ind) in enumerate(split_gen):
        file_ind = file_ind[0]

        x = np.squeeze(batch_x, axis=0)

        y_probs = make_prediction_img(
            x, options.target_size[0],
            lambda x: predict_x(x, model))
        
        y_preds = dataset.one_hot_to_rgb_batch(y_probs)
        prediction_file_path = join(
            predictions_path,
            generator.dataset.get_output_file_name(file_ind))
        
        print('save: %s' % (prediction_file_path))
        save_img(y_preds, prediction_file_path)

def validation_predict(run_path, model, options, generator):
    predict(run_path, model, options, generator, VALIDATION)

def test_predict(run_path, model, options, generator):
    predict(run_path, model, options, generator, TEST)
