from mmengine.evaluator import BaseMetric
from mmengine import EVALUATOR, METRICS
from mmengine.visualization import Visualizer
import torch
import random
from matplotlib import pyplot as plt

@METRICS.register_module()
class ClassifyAccuracy(BaseMetric):
    def process(self, data, preds) -> None:
        self.results.append(((data['data_samples']['target'] == torch.Tensor(preds)).sum(), len(preds)))
        # # 可视化
        # if not getattr(self, 'visual_status', False):
        #     if random.random() > 0.8:
        #         self.visual(data, preds)
        #         self.visual_status = True

    def compute_metrics(self, results):
        correct, all_iters = zip(*results)
        acc = sum(correct) / sum(all_iters)

        return dict(accuracy=acc)

    def visual(self, data, preds):
            
            visual_list = []
            visualizer = Visualizer.get_current_instance()
            wandb = visualizer.get_backend('WandbVisBackend').experiment
            for i in range(len(preds)):
                # set_image bugs , will auto astype uint8, it make 0-1 image convert to all 0(black)
                visualizer.set_image(data['input'][i].permute(1,2,0).cpu().numpy()*255.0)
                # visualizer.draw_texts(f"{1}:{1}", torch.tensor([0, 0]))
                save_img = visualizer.get_image()
                gt_index = data['data_samples']['target'][i].cpu().item()
                pred_index = preds[i]
                if self.dataset_meta is not None and \
                    'classes' in self.dataset_meta and \
                        len(self.dataset_meta['classes']) > 0:
                    classes = self.dataset_meta['classes']
                    caption = f"GT:{classes[gt_index]} | Pred:{classes[pred_index]}"
                else:
                    caption = f"gt-{gt_index} | pred-{pred_index}"
                image = wandb.Image(save_img, caption=caption)
                visual_list.append(image)
            # visualizer.set_image was only call when single image, 
            # so we need to call wandb log directly
            wandb.log({'log an input and target': visual_list})
            

