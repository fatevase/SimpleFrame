import os.path as osp
import cv2
from typing import List
from mmengine.dataset import BaseDataset


class CvDDataset(BaseDataset):


    def filter_data(self) -> List[dict]:
        data_list = []
        def lisMap(d):
            return dict(img_path=d[0], img_label=d[0])
        for d in self.data_list:
            d = lisMap(d)
            d = self.parse_data_info(d)
            data_list.append(d)
        # data_list = list(map(lisMap, self.data_list))
        return data_list
        

    def load_data_list(self):
        metainfo = dict(classes=('cat', 'dog'))
        data_list = []
        with open(osp.join(self.data_root, self.ann_file), 'r') as annos:
            anno = annos.readline().strip().split(' ')
            
            # data_info = self.parse_data_info(dict(img_path=anno[0], img_label=anno[1]))
            data_list.append(anno)
        for k, v in metainfo.items():
            self._metainfo.setdefault(k, v)

        return data_list

    


class LoadImage:

    def __call__(self, results):
        results['img'] = cv2.imread(results['img_path'])
        print(results['img_path'])
        return results

class ParseImage:

    def __call__(self, results):
        results['img_shape'] = results['img'].shape
        return results

pipeline = [
    LoadImage(),
    ParseImage(),
]

if __name__ == "__main__":

    cvd_loader = CvDDataset(
        data_root='/Users/vase/Downloads/DogsVsCats/',
        data_prefix=dict(img_path='imgs/'),
        ann_file='train.txt',
        pipeline=pipeline)
    print(cvd_loader.metainfo)
    # dict(classes=('cat', 'dog'))

    print(cvd_loader.get_data_info(0))

    print(next(iter(cvd_loader)))