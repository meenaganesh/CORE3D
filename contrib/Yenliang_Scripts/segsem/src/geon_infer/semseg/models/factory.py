from os.path import isfile, join
from subprocess import call

from geon_infer.semseg.models.fcn_resnet import make_fcn_resnet
from geon_infer.semseg.models.fcn_resnet_depth import make_fcn_resnet_depth



def make_model(options, generator):
    model_type = options.model_type
    input_shape = (options.target_size[0], options.target_size[1],
                   len(options.active_input_inds))
    nb_labels = generator.dataset.nb_labels

    if model_type == "fcn_resnet":
        model = make_fcn_resnet(
            input_shape, nb_labels, options.use_pretraining,
            options.freeze_base)
    
    elif model_type == "fcn_resnet_depth":
        model = make_fcn_resnet_depth(
            input_shape, nb_labels, options.use_pretraining,
            options.freeze_base)
    else:
        raise ValueError('{} is not a valid model_type'.format(model_type))

    return model

def load_model(run_path, options, generator):
    model = make_model(options, generator)
    file_name = options.save_model
    model_path = join(run_path, file_name)
    
    print('loading model weights: %s' % (model_path))
    model.load_weights(model_path, by_name=True)
    return model


def get_model(run_path, options, generator):
    model_path = join(run_path, options.save_model)

    if isfile(model_path):
        model = load_model(run_path, options, generator)
        print('Continuing training from saved model.')
    else:
        model = make_model(options, generator)
        print('Creating new model.')

    return model
