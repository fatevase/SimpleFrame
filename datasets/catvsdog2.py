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
class CvDDataset(ProxyDataset):

    def __init__(self, ann_list:List=[], **kwargs):
        super(CvDDataset, self).__init__(ann_list=ann_list, **kwargs)

    def _filterParentArgs(self, kwargs):
        self.ann_list = kwargs.pop('ann_list',[]) # change ann_file to ann_list for multi annotation file
        return kwargs

    def _loadAnnotations(self) -> Dict:
        metainfo = dict(classes=('cat', 'dog'), ann_list=self.ann_list)
        data_list = []
        mode_str ='Val' if self.test_mode else 'Train' 

        for ann_file in self.ann_list:
            rich_args = dict(file=osp.join(self.data_root, ann_file),
                    mode='r', description=f":floppy_disk: DL {mode_str}:")
            with open(**rich_args) as file:  # type: ignore
                annos = file.readlines()
            loop = track(annos, description=f":hammer: DP {mode_str}:")
            for anno in loop:
                anno = anno.strip().split(' ')
                data_list.append(dict(img_path1=anno[0], img_label=anno[1]))
                time.sleep(0.0001)
        return dict(metainfo=metainfo, data_list=data_list)


class ReInDict:
    def __init__(self, **kwargs):
        self.remap = kwargs
    def __call__(self, results):
        for key, value in self.remap.items():
            results[value] = results.pop(key)
        return results

class LoadImage:

    def __call__(self, results):
        results['img'] = cv2.imread(results['img_path'])
        return results

class ParseImage:
    def __call__(self, results):
        results['img_shape'] = results['img'].shape
        return results

class ReOutDict:
    def __init__(self, **kwargs):
        self.remap = kwargs
    def __call__(self, results):
        new_result = dict()
        for key, value in self.remap.items():
            new_result[value] = results.pop(key)
        new_result['metainfo'] = results
        return new_result

pipeline = [
    LoadImage(),
    ParseImage(),
]

if __name__ == "__main__":

    # cvd_loader = CvDDataset(
    #     data_root='/Users/vase/Downloads/DogsVsCats/',
    #     data_prefix=dict(img_path1=dict(new_key='img_path', path_prefix='imgs/')),
    #     ann_file='train.txt',
    #     pipeline=pipeline,)

    from mmengine import Registry
    cfg=dict(
    type='CvDDataset',
    data_root='/Users/vase/Downloads/DogsVsCats/',
    data_prefix=dict(img_path1='imgs/'),
    ann_list=['train.txt'],
    pipeline=[ReInDict(img_path1='img_path'), LoadImage(), ParseImage(), ReOutDict(img='input', img_label='target')])

    cvd_loader = mmengine.build_from_cfg(cfg, DATASETS)
    # handle parent init if  lazy_load=True
    # cvd_loader.full_init()
    print(cvd_loader.metainfo)
    # print(cvd_loader.get_data_info(0))
    print(next(iter(cvd_loader)))