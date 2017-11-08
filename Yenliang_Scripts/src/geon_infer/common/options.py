import pdb
class Options():

    def __init__(self, options):
        if 'train_stages' in options and options['train_stages'] is not None:
            train_stages = options['train_stages']
            options.update(train_stages[0])
        self.train_stages = options.get('train_stages')

        # Required options
        self.model_type = options['model_type']
        self.run_name = options['run_name']
        self.dataset_name = options['dataset_name']
        self.generator_name = options['generator_name']
        self.batch_size = options['batch_size']
        self.epochs = options['epochs']
        self.steps_per_epoch = options['steps_per_epoch']
        self.validation_steps = options['validation_steps']
        self.active_input_inds = options['active_input_inds']
        self.dataset_path = options['dataset_path'] 
        self.result_path = options['result_path']
        self.save_model = options['save_model']

        # Optional options
        self.target_size = options.get('target_size', (256, 256))
        self.optimizer = options.get('optimizer', 'adam')
        self.init_lr = options.get('init_lr', 1e-3)
        self.patience = options.get('patience')
        self.lr_schedule = options.get('lr_schedule')
        self.train_ratio = options.get('train_ratio')
