from typing import Dict, List, Union
from mmengine.dataset import BaseDataset
import os.path as osp
import abc
from mmengine.utils import is_abs

class ProxyDataset(BaseDataset):
    """ override
        do not serialize data_list if data_list include numpy or some huge data.
        if _loadAnnotations return data_list include a lot of data,
        make sure serialize_data=False
    """

    def __init__(self, **kwargs):
        kwargs = self._filterParentArgs(kwargs)
        super(ProxyDataset, self).__init__(**kwargs)
        # self.annotations = self._loadAnnotations() if self.data_list is None else self.data_list


    @abc.abstractmethod
    def _filterParentArgs(self, args) -> Dict:
        """Handle parent arguments.
        Returns:
            dict: Parent arguments.
        """
        args['ann_list'] = args.get('ann_list', [])
        args['data_prefix'] = args.get('data_prefix', {})      
        return args

    # need complete the function on Subclass
    @abc.abstractmethod
    def _loadAnnotations(self) -> Dict:
        """Load annotations on subclass
            was a dict dict(metainfo=dict(...), data_list=list(...))
            for common dataset, data_list is a list of dict
            core need include input(img_path, seq_path) and target(img_label) for training and evaluation
            we also need to add some other information for data_list, such as img_shape, img_size, etc.
        """
        return dict()


    def _serialize_data(self):
        """ override
        """
        import numpy as np
        import pickle
        import gc

        def _serialize(data):
            # TODO: need fixed for dump large tensor array if data include tensor array
            # BUG: dump large tensor array will cause memory leak.
            # pickle focus on python data type.
            buffer = pickle.dumps(data, protocol=4)
            return np.frombuffer(buffer, dtype=np.uint8)


        data_list = [_serialize(x) for x in self.data_list]
        address_list = np.asarray([len(x) for x in data_list], dtype=np.int64)
        data_address: np.ndarray = np.cumsum(address_list)

        # TODO Check if np.concatenate is necessary
        data_bytes = np.concatenate(data_list)
        # Empty cache for preventing making multiple copies of
        # `self.data_info` when loading data multi-processes.
        self.data_list.clear()
        gc.collect()
        return data_bytes, data_address


    def _join_prefix(self) -> None:
        """ override data_prefix
            handler data_prefix dict from dataset 
            concat data_root and path_prefix
        """
        for i in range(len(self.ann_list)):
            if not is_abs(self.ann_list[i]) and self.ann_list[i]:
                self.ann_list[i] = osp.join(self.data_root, self.ann_list[i])
        # Automatically join data directory with `self.root` if path value in
        # `self.data_prefix` is not an absolute path.
        for data_key, prefix in self.data_prefix.items():
            if isinstance(prefix, str):
                if not is_abs(prefix):
                    prefix = osp.join(self.data_root, prefix)
            elif isinstance(prefix, dict):
                assert 'path_prefix' in prefix, (
                    'path_prefix must be in data_prefix')
                prefix['path_prefix'] = osp.join(self.data_root, prefix['path_prefix'])
            
            self.data_prefix[data_key] = prefix
    

    def parse_data_info(self, raw_data_info: dict) -> Union[dict, List[dict]]:
        """Parse data information to target format.
            concat annotation file path and prefix_path;
            rename key for data_list;
            override function also can change the raw_data_info key for next part
        Args:
            data_info (dict): single Data information loaded from annotation.
        Returns:
            dict: Parsed data information.
        """
        for prefix_key, prefix in self.data_prefix.items():
            assert prefix_key in raw_data_info, (
                f'raw_data_info: {raw_data_info} dose not contain prefix key'
                f'{prefix_key}, please check your data_prefix.')
            if isinstance(prefix, dict):
                raw_data_info[prefix_key] = osp.join(prefix['path_prefix'],
                                                 raw_data_info[prefix_key])
                raw_data_info[prefix['new_key']] = raw_data_info.pop(prefix_key)
            elif isinstance(prefix, str):
                raw_data_info[prefix_key] = osp.join(prefix,
                                                 raw_data_info[prefix_key])
        return raw_data_info

    def load_data_list(self) -> List[dict]:
        """ Override this function to load data_list
        Returns:
            list[dict]: A list of annotation.
        """
        
        # if lazy loading is close, need it
        # if not hasattr(self, 'annotations'):
        #     self.annotations = self._loadAnnotations()

        annotations = self._loadAnnotations()
        if not isinstance(annotations, dict):
            raise TypeError(f'The annotations loaded from annotation file '
                            f'should be a dict, but got {type(annotations)}!')
        if 'data_list' not in annotations or 'metainfo' not in annotations:
            raise ValueError('Annotation must have data_list and metainfo '
                             'keys')
        metainfo = annotations['metainfo']
        raw_data_list = annotations['data_list']

        # Meta information load from annotation file will not influence the
        # existed meta information load from `BaseDataset.METAINFO` and
        # `metainfo` arguments defined in constructor.
        for k, v in metainfo.items():
            self._metainfo.setdefault(k, v)

        # load and parse data_infos.
        data_list = []


        for raw_data_info in raw_data_list:
            # parse raw data information to target format
            data_info = self.parse_data_info(raw_data_info)
            if isinstance(data_info, dict):
                # For nonsequence tasks, `data_info` should information if single
                # image, such as dict(img_path='xxx', width=360, ...)
                data_list.append(data_info)
            elif isinstance(data_info, list):
                # For continuous input tasks, `data_info` could contain image
                # information of multiple frames, such as
                # [dict(video_path='xxx', timestamps=...),
                #  dict(video_path='xxx', timestamps=...)]
                for item in data_info:
                    if not isinstance(item, dict):
                        raise TypeError('data_info must be list of dict, but '
                                        f'got {type(item)}')
                data_list.extend(data_info)
            else:
                raise TypeError('data_info should be a dict or list of dict, '
                                f'but got {type(data_info)}')
        return data_list
    
    