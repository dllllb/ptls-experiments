from ptls.frames.coles.coles_module import CoLESModule
from ptls.nn.seq_encoder.containers import SeqEncoderContainer

class CoLESModuleWarmup(CoLESModule):
    def __init__(self,
                 seq_encoder: SeqEncoderContainer = None,
                 head=None,
                 loss=None,
                 validation_metric=None,
                 optimizer_partial=None,
                 lr_scheduler_partial=None,
                 warmup_steps = 500,
                 initial_lr = 0.001):
        
        super().__init__(seq_encoder,
                         head,
                         loss,
                         validation_metric,
                         optimizer_partial,
                         lr_scheduler_partial)
        self.warmup_steps = warmup_steps
        self.initial_lr = initial_lr
        
    def optimizer_step(self, 
                       epoch, 
                       batch_idx, 
                       optimizer, 
                       optimizer_idx, 
                       optimizer_closure, 
                       on_tpu=False, 
                       using_native_amp=False, 
                       using_lbfgs=False):
        
        optimizer.step(closure = optimizer_closure)
        if self.trainer.global_step < self.warmup_steps:
            lr_scale = min(1., float(self.trainer.global_step + 1) / self.warmup_steps)
            for pg in optimizer.param_groups:
                pg['lr'] = lr_scale * self.initial_lr
        

        
