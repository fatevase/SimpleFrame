import os.path as osp
import cv2
from typing import Dict, List, Literal
from mmengine import DATASETS
import mmengine
import numpy as np
from rich.progress import open, track, wrap_file
import time
import pickle

from .proxy_dataset import ProxyDataset
from utils.rich_utils import processInfo

@DATASETS.register_module()
class Cifar10Dataset(ProxyDataset):

    def __init__(self, data_root='', ann_list:List=[], **kwargs):
        super(Cifar10Dataset, self).__init__(data_root=data_root,ann_list=ann_list, **kwargs)

    def _filterParentArgs(self, args):
        self.ann_list = args.pop('ann_list',[]) # change ann_file to ann_list for multi annotation file
        args['data_prefix'] = args['data_prefix'] if 'data_prefix' in args else {}
        return args

    def _loadAnnotations(self) -> Dict:
        # cifar 10 classes
        metainfo = dict(
            classes=['airplane', 'automobile', 'bird', 'cat', 
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck'],
            ann_list=self.ann_list
        )
        data_list = []
        mode_str ='Val' if self.test_mode else 'Train'

        for ann_file in self.ann_list:
            rich_args = dict(file=osp.join(self.data_root, ann_file),
                    mode='rb', description=processInfo(f"DL"))
            with open(**rich_args) as file:  # type: ignore
                annos = pickle.load(file, encoding='bytes')

            zip_annos = zip(annos[b'labels'], annos[b'data'], annos[b'filenames'])
            loop = track(zip_annos, 
                description=processInfo(f"DP {annos[b'batch_label'].decode('utf-8')}:"),
                total=len(annos[b'labels']),
            )
            for anno in zip(loop):
                anno = anno[0]
                data_list.append(
                    dict(img=anno[1].astype(np.float32).reshape(3, 32, 32).transpose(1,2,0)/255.0,
                        img_label=anno[0], name=anno[2].decode('utf-8')
                    )
                )
            
        return dict(metainfo=metainfo, data_list=data_list)


if __name__ == "__main__":

    # cvd_loader = CvDDataset(
    #     data_root='/Users/vase/Downloads/DogsVsCats/',
    #     data_prefix=dict(img_path1=dict(new_key='img_path', path_prefix='imgs/')),
    #     ann_file='train.txt',
    #     pipeline=pipeline,)

    from mmengine import Registry
    cfg=dict(
    type='Cifar10Dataset',
    data_root='/Users/vase/Downloads/cifar-10/',
    # data_prefix=dict(),
    ann_list=['data_batch_1'],
    pipeline=[])

    cvd_loader = mmengine.build_from_cfg(cfg, DATASETS)
    # handle parent init if  lazy_load=True
    # cvd_loader.full_init()
    print(cvd_loader.metainfo)
    # print(cvd_loader.get_data_info(0))
    get_data = next(iter(cvd_loader))
    print(get_data)
    cv2.imshow('img', get_data['img'][...,::-1])
    cv2.waitKey(0)