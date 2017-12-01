from os.path import join, isfile
import math

from keras.callbacks import (Callback, ModelCheckpoint, CSVLogger,
                             ReduceLROnPlateau, LambdaCallback,
                             LearningRateScheduler)
from keras.optimizers import Adam, RMSprop

from ..data.generators import TRAIN, VALIDATION
from geon_infer.common.utils import _makedirs
import pdb

ADAM = 'adam'
RMS_PROP = 'rms_prop'
TRAIN_MODEL = 'train_model'

def make_callbacks(run_path, options, log_path):
    model_checkpoint = ModelCheckpoint(
        filepath=join(run_path, options.save_model), period=1, save_weights_only=True)
   
    logger = CSVLogger(log_path, append=True)
    
    callbacks = [model_checkpoint, logger]

    #if options.patience:
    #    callback = ReduceLROnPlateau(
    #        verbose=1, epsilon=0.001, patience=options.patience)
    #    callbacks.append(callback)

    #if options.lr_schedule:
    #    def get_lr(epoch):
    #        for epoch_thresh, lr in options.lr_schedule:
    #            if epoch >= epoch_thresh:
    #                curr_lr = lr
    #            else:
    #                break
    #        return curr_lr
    #    callback = LearningRateScheduler(get_lr)
    #    callbacks.append(callback)

    return callbacks


def get_initial_epoch(log_path):
    initial_epoch = 0
    if isfile(log_path):
        with open(log_path) as log_file:
            line_ind = 0
            for line_ind, _ in enumerate(log_file):
                pass
            initial_epoch = line_ind

    return initial_epoch


def get_lr(epoch, lr_schedule):
    for epoch_thresh, lr in lr_schedule:
        if epoch >= epoch_thresh:
            curr_lr = lr
        else:
            break
    return curr_lr


def train_model(run_path, model, options, generator):
    print(model.summary())

    print('train generator')
    train_gen = generator.make_split_generator(
        TRAIN, model_type=options.model_type, target_size=options.target_size, batch_size=options.batch_size,
        shuffle=True, augment=False, normalize=False)
      
    if options.optimizer == ADAM:
        optimizer = Adam(lr=options.init_lr)
    
    elif options.optimizer == RMS_PROP:
        optimizer = RMSprop(lr=options.init_lr)

    print('compiling')    
    print('model type: %s' % (options.model_type))

    if options.model_type == "fcn_resnet": 
      model.compile(optimizer,loss='categorical_crossentropy',metrics=['accuracy'])
    
    elif options.model_type == "fcn_resnet_depth":
      model.compile(optimizer,loss='mean_squared_error',metrics=['mse'])
      
    log_path = join(run_path, 'log.txt')
    initial_epoch = get_initial_epoch(log_path)

    callbacks = make_callbacks(run_path, options, log_path) 
    
    model.fit_generator(
            train_gen,
            initial_epoch=initial_epoch,
            steps_per_epoch=options.steps_per_epoch,
            epochs=options.epochs,
            callbacks=callbacks)
        
