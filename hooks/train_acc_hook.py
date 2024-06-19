import torch
import torch.nn as nn
from mmengine.hooks import Hook
from mmengine import HOOKS

@HOOKS.register_module()
class TrainAccuracyHook(Hook):
    def after_train_epoch(self, runner):
        runner.model.eval()
        
        with torch.no_grad():
            runner.model.eval()
            for idx, data_batch in enumerate(runner.train_dataloader):
                runner._val_loop.run_iter(idx, data_batch)

        # compute metrics
        metrics = runner._val_loop.evaluator.evaluate(len(runner.train_dataloader.dataset))
        runner.logger.info(f"Train Accuracy: {metrics['accuracy']}")
        runner.model.train()
        return metrics