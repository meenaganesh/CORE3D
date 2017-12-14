from geon_infer.common.options import Options

class SemsegOptions(Options):

 def __init__(self, options):
      Options.__init__(self,options)

      self.nb_eval_samples = options.get('nb_eval_samples')
      self.use_pretraining = options['use_pretraining']
      self.freeze_base = options['freeze_base']
      self.eval_target_size = options.get('eval_target_size')
