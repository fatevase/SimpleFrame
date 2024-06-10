import os.path as osp
import cv2
from typing import Dict, List, Literal
from proxy_dataset import ProxyDataset
from mmengine import DATASETS
import mmengine
import rich
from rich.progress import open, track, wrap_file
import time

@DATASETS.register_module()
class FSS1000Dataset(ProxyDataset):

    def __init__(self, ann_list:List=[], **kwargs):
        super(FSS1000Dataset, self).__init__(ann_list=ann_list, **kwargs)

    def _filterParentArgs(self, kwargs):
        self.ann_list = kwargs.pop('ann_list',[]) # change ann_file to ann_list for multi annotation file
        return kwargs

    def _loadAnnotations(self) -> Dict:
        metainfo = dict(classes=[], ann_list=self.ann_list)
        data_list = []

        for ann_file in self.ann_list:
            rich_args = dict(file=osp.join(self.data_root, ann_file),
                    mode='r', description=f":floppy_disk: DL :")
            with open(**rich_args) as file:  # type: ignore
                annos = file.readlines()[1:]
            loop = track(annos, description=f":hammer: DP :")
            for anno in loop:
                anno = anno.strip().split(',')
                if anno[3] not in metainfo['classes']:
                    metainfo['classes'].append(anno[3])
                anno[3] = metainfo['classes'].index(anno[3])
                data_list.append(
                    dict(img_path=anno[1], img_label=anno[3], mask_path=anno[2],
                        bbox=dict(x_min=float(anno[4]), y_min=float(anno[5]), 
                            x_max=float(anno[6]), y_max=float(anno[7]))
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
    type='FSS1000Dataset',
    data_root='/Users/vase/Downloads/FSS1000/',
    data_prefix=dict(img_path='FSS-1000/'),
    ann_list=['fss-1000.csv'],
    pipeline=[])

    cvd_loader = mmengine.build_from_cfg(cfg, DATASETS)
    # handle parent init if  lazy_load=True
    # cvd_loader.full_init()
    print(cvd_loader.metainfo)
    # print(cvd_loader.get_data_info(0))
    print(next(iter(cvd_loader)))