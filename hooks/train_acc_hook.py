import torch
import torch.nn as nn
from mmengine.hooks import Hook
from mmengine import HOOKS

@HOOKS.register_module()
class TrainAccuracyHook(Hook):
    def after_train_epoch(self, runner):
        runner.model.eval()
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in runner.dataloader:
                inputs, labels = batch['input'], batch['target']
                outputs = runner.model(inputs)
                
                preds = outputs.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        metrics = runner.evaluator.evaluate(all_preds, all_labels)
        runner.logger.info(f"Train Accuracy: {metrics['accuracy']}")
        runner.model.train()