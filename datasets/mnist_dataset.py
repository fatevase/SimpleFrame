import os.path as osp
import cv2
from typing import Dict, List
from .proxy_dataset import ProxyDataset
from mmengine import DATASETS
import mmengine
from rich.progress import open, track, wrap_file
from torchvision import datasets
import numpy as np

@DATASETS.register_module()
class MMNISTDataset(ProxyDataset):

    def __init__(
        self, root="MNIST", download=True, **kwargs):
        super(MMNISTDataset, self).__init__(
            root=root, download=download, **kwargs)

    def _filterParentArgs(self, args) -> Dict:
        self.download = args.pop('download', True)
        self.root = args.pop('root', 'MMIST')
        return super()._filterParentArgs(args)

    def _loadAnnotations(self) -> Dict:
        metainfo = dict(classes=('0', '1', '2', '3', '4', '5', '6', '7', '8', '9'))
        data_list = []
        mode_str ='Val' if self.test_mode else 'Train'
        dataset = datasets.MNIST(root=self.root, download=self.download, train=not self.test_mode)

        input, target = dataset.data, dataset.targets
        loop = track(zip(input, target), description=f":floppy_disk: DL {mode_str}:", total=len(input))
        for i, t in loop:
            # BUG: this bug was cause by BaseDataset._serialize_data
            data_list.append(dict(img=i.numpy().astype(np.float32)/255, img_label=int(t)))
        return dict(metainfo=metainfo, data_list=data_list)


if __name__ == "__main__":

    cfg=dict(
        type='MMNISTDataset',
        root='MNIST',
        download=True,
        test_mode=False,
        # serialize_data=False,
    )

    cvd_loader = mmengine.build_from_cfg(cfg, DATASETS)
    # defeatrue lazy_laod
    # cvd_loader.full_init()
    print(cvd_loader.metainfo)
    # print(cvd_loader.get_data_info(0))
    next_data = next(iter(cvd_loader))
    print(next_data['img'])