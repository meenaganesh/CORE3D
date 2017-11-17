from os.path import join
import sys

from geon_infer.common.utils import setup_run
from geon_infer.common.tasks.plot_curves import plot_curves, PLOT_CURVES

from .options import SemsegOptions
from .data.factory import get_data_generator
from .models.factory import get_model, load_model
from .tasks.train_model import train_model, TRAIN_MODEL
from .tasks.validation_eval import validation_eval, VALIDATION_EVAL
from .tasks.predict import (validation_predict, test_predict, VALIDATION_PREDICT, TEST_PREDICT)

valid_tasks = [TRAIN_MODEL, PLOT_CURVES, VALIDATION_PREDICT, VALIDATION_EVAL, TEST_PREDICT]

import pdb

def run_tasks(options_dict, tasks):
    options = SemsegOptions(options_dict)
    generator = get_data_generator(options)
    
    run_path = join(options.result_path, options.run_name)
    setup_run(run_path)

    if len(tasks) == 0:
        tasks = valid_tasks

    for task in tasks:
        if task not in valid_tasks:
            raise ValueError('{} is not a valid task'.format(task))

    for task in tasks:
        if task == TRAIN_MODEL:
            
            model = get_model(
                run_path, options, generator)
            
            print('epochs: %s learning rate: %f' % (options.epochs, options.init_lr))
            train_model(run_path, model, options, generator)
             
            if options.train_stages:
                for stage in options.train_stages[1:]:
                    for key, value in stage.items():
                        if key == 'epochs':
                            options.epochs += value
                        else:
                            setattr(options, key, value)
                    
                    print('epochs: %s learning rate: %f' % (options.epochs, options.init_lr))
                    model = get_model(
                        run_path, options, generator)
                    
                    train_model(
                        run_path, model, options, generator)
        
        elif task == PLOT_CURVES:
            plot_curves(run_path)
        
        elif task == VALIDATION_EVAL:
            options.use_pretraining = False
            
            model = load_model(run_path, options, generator)
            
            validation_eval(run_path, model, options, generator)
        
        elif task == TEST_PREDICT:
           
            options.use_pretraining = False
            model = load_model(
                run_path, options, generator)
            
            test_predict(run_path, model, options, generator)
        
        elif task == VALIDATION_PREDICT:
            
            options.use_pretraining = False
            model = load_model(
                run_path, options, generator)
            
            validation_predict(run_path, model, options, generator) 

