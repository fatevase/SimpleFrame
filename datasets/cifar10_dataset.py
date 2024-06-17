import os.path as osp
import cv2
from typing import Dict, List, Literal
from mmengine import DATASETS
import mmengine
import numpy as np
from rich.progress import open, track, wrap_file
import time
import pickle
import os
import io # only for write file though rich open override orin and cant write
from .proxy_dataset import ProxyDataset
from utils.rich_utils import processInfo

@DATASETS.register_module()
class Cifar10Dataset(ProxyDataset):

    def __init__(self, data_root='', ann_list:List=[], download=False, **kwargs):
        super(Cifar10Dataset, self).__init__(data_root=data_root, ann_list=ann_list, download=download, **kwargs)

    def _filterParentArgs(self, args):
        self.ann_list = args.pop('ann_list',[]) # change ann_file to ann_list for multi annotation file
        args['data_prefix'] = args['data_prefix'] if 'data_prefix' in args else {}

        keys_to_remove = [key for key in args.keys() if key in ['download']]
        for key in keys_to_remove:
            setattr(self, key, args.pop(key))
        return args

    def _loadAnnotations(self) -> Dict:
        # cifar 10 classes

        self._check_and_download_data()

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


    def _check_and_download_data(self):
        """
        Check if CIFAR-10 data exists, and if not, download it.
        """
        if not self.download:
            return
        for ann_file in self.ann_list:
            if not os.path.exists(osp.join(self.data_root, ann_file)):
                print("CIFAR-10 dataset not found. Downloading...")
                self._download_data()
                break

    def _download_data(self):
        """
        Download and extract the CIFAR-10 dataset.
        """
        CIFAR10_FILENAME = "cifar-10-python.tar.gz"
        CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
        os.makedirs(self.data_root, exist_ok=True)
        tar_path = osp.join(self.data_root, CIFAR10_FILENAME)
        import requests
        import tarfile
        # Download the dataset
        # Remove existing incomplete file
        if os.path.exists(tar_path):
            os.remove(tar_path)
        with requests.get(CIFAR10_URL, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            with io.open(tar_path, 'wb') as f:
                for chunk in track(
                        r.iter_content(chunk_size=8192),
                        description="Downloading CIFAR-10",
                        total=total_size // 8192,
                        show_speed=True):
                    if chunk:  f.write(chunk)
        
        # Extract the tar file
        print("Extracting CIFAR-10 dataset...")
        with tarfile.open(tar_path, 'r:gz') as tar:
            members = []
            for member in tar.getmembers():
                if member.name.startswith('cifar-10-batches-py/'):
                    # 去掉目录名前缀
                    member.name = member.name[len('cifar-10-batches-py/'):]
                    # 提取文件到目标目录
                    if member.name:  # 避免提取空的目录名
                        members.append(member)

            tar.extractall(self.data_root, members)

        # Clean up the tar file
        os.remove(tar_path)
        print("CIFAR-10 dataset downloaded and extracted successfully.")

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