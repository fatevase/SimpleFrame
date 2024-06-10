from mmengine.hooks import Hook
from mmengine import HOOKS
from mmengine.visualization import Visualizer
import numpy as np


@HOOKS.register_module()
class ModelVisHook(Hook):
    def before_run(self, runner):
        model = runner.model
    
        input_hook_data = {}
        output_hook_data = {}
        def get_activation(name):
            def hook(model,input,output):
                self.input_hook_data[name] = input[0].detach() # input type is tulple, only has one element, which is the tensor
                self.output_hook_data[name] = output.detach()  # output type is tensor
            return hook
        self.loss_hook = model.loss.register_forward_hook(get_activation('loss_hook'))
        self.pred_hook = model.cnn_decoder.register_forward_hook(get_activation('pred_hook'))
        self.input_hook_data = input_hook_data
        self.output_hook_data = output_hook_data
        self.epoch_loss = 0.0
        self.visual_data = dict(data=dict(), preds=list(), epoch=0)
        self.val_visual_step = 1


    # def after_train_iter(self,runner,batch_idx: int,
    #                       data_batch=None,outputs = None) -> None:
    #     if batch_idx == self.random_batch:
    #         preds = self.output_hook_data['pred_hook'].argmax(dim=1)
    #         data_meta = runner.visualizer.dataset_meta
    #         self.visual(data_batch, preds, data_meta)

    # def before_train_epoch(self, runner):
    #     # random choose one image to show
    #     # got random batch index
    #     # random chose a list random iters
    #     epoch_iters = runner.max_iters // runner.max_epochs
    #     self.random_batch = np.random.choice(range(epoch_iters))


    def before_val_epoch(self, runner) -> None:
        epoch_iters = len(runner.val_dataloader) // runner.val_dataloader.batch_size
        self.random_batch = np.random.choice(range(epoch_iters))
    
    def after_val_iter(self, runner, batch_idx: int,
                        data_batch = None, outputs = None) -> None:
        if batch_idx == self.random_batch:
            # preds = self.output_hook_data['pred_hook'].argmax(dim=1)
            preds = outputs
            data_meta = runner.visualizer.dataset_meta
            self.visual(data_batch, preds, data_meta, self.val_visual_step)
            # self.visualTable(data_batch, preds, data_meta, self.val_visual_step)
            self.val_visual_step += 1

    # def after_train_iter(self, runner, batch_idx, data_batch, outputs):
        # if not outputs: return
        # outputs = outputs
        # loss_result = self.input_hook_data['loss_hook'] # pred result
        # visualizer = Visualizer.get_current_instance()
        # wandb = visualizer.get_backend('WandbVisBackend').experiment

        # visual_list = []
        # self.epoch_loss += outputs['loss'].item()
        # if self.end_of_epoch(runner.train_dataloader, batch_idx):
        #     self.epoch_loss /= runner.max_iters
        #     wandb.log({'loss': self.epoch_loss, 'epoch': runner.epoch})
        #     self.epoch_loss = 0.0
    
    def visual(self, data, preds, dataset_meta=None, epoch=1):
        visual_list = []
        visualizer = Visualizer.get_current_instance()
        wandb = visualizer.get_backend('WandbVisBackend').experiment
        for i in range(len(preds)):
            # set_image bugs , will auto astype uint8, it make 0-1 image convert to all 0(black)
            # visualizer.set_image(data['input'][i].permute(1,2,0).cpu().numpy()*255.0)
            # visualizer.draw_texts(f"{1}:{1}", torch.tensor([0, 0]))
            save_img = data['input'][i].permute(1,2,0).cpu().numpy()
            gt_index = data['data_samples']['target'][i].cpu().item()
            pred_index = preds[i]
            caption = self.pred2caption(dataset_meta, pred_index, gt_index)
            image = wandb.Image(save_img, caption=caption)
            visual_list.append(image)
        # step only be setting for current or next step
        # cant set step like 1,2,3,... ,， so we need additonal variable save in it
        wandb.log({f"Pred-Result": visual_list})

    def pred2caption(self, data_meta, pred, gt):
        caption = f"GT-{gt} | Pred-{[pred]}"
        if not data_meta or 'classes' not in data_meta:
            return caption
        classes = data_meta['classes']
        if len(classes) <= pred or len(classes) <= gt:
            return caption
        caption = f"GT:{classes[gt]} | Pred:{classes[pred]}"
        return caption

    def visualTable(self, data_batch, preds, data_meta, visual_step):
        visualizer = Visualizer.get_current_instance()
        wandb = visualizer.get_backend('WandbVisBackend').experiment
        data = {'img':[], 'caption':[], 'visual_step':[], 'preds':[], 'truth':[]}

        # data['id'] = [i for i in range(len(preds))]
        data['preds'] = preds.cpu().numpy().tolist()
        data['truth'] = data_batch['data_samples']['target'].cpu().numpy().tolist()
        for i in range(len(preds)):
            save_img = data_batch['input'][i].permute(1,2,0).cpu().numpy()
            data['img'].append(wandb.Image(save_img))
        data['caption'] = [self.pred2caption(data_meta, pred, gt) for pred, gt in zip(preds, data['truth'])]
        data['visual_step'] = [visual_step] * len(preds)
        self.wandbLogTable(wandb, data)


    
    def wandbLogTable(self, wandb, table):
        # ["id", "image", "guess", "truth"]
        columns = list(table.keys())
        num_rows = len(table[list(table.keys())[0]])
        wandb_table = wandb.Table(columns=columns)
        for i in range(num_rows):
            row = [table[column][i] for column in table.keys()]
            wandb_table.add_data(*row)
        wandb.log({"Pred-Result": wandb_table})
