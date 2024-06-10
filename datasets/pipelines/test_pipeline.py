from mmengine import TRANSFORMS
import cv2
import numpy as np
import torch

@TRANSFORMS.register_module()
class ReInDict:
    def __init__(self, **kwargs):
        self.remap = kwargs
    def __call__(self, results):
        for key, value in self.remap.items():
            results[value] = results.pop(key)
        return results



@TRANSFORMS.register_module()
class ToTensor(object):
    """Convert some results to :obj:`torch.Tensor` by given keys.

    Args:
        keys (Sequence[str]): Keys that need to be converted to Tensor.
    """
    def to_tensor(self, data) -> torch.Tensor:
        """Convert objects of various python types to :obj:`torch.Tensor`.

        Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
        :class:`Sequence`, :class:`int` and :class:`float`.

        Args:
            data (torch.Tensor | numpy.ndarray | Sequence | int | float): Data to
                be converted.
        """
        if isinstance(data, torch.Tensor):
            return data
        elif isinstance(data, np.ndarray):
            return torch.from_numpy(data)
        elif isinstance(data, int):
            return torch.LongTensor([data])
        elif isinstance(data, float):
            return torch.FloatTensor([data])
        else:
            raise TypeError(f'type {type(data)} cannot be converted to tensor.')

    def __init__(self, keys=['img'], cfg=None):
        self.keys = keys

    def __call__(self, sample):
        data = sample.copy()
        if len(sample['img'].shape) < 3:
            sample['img'] = np.expand_dims(sample['img'], -1)
        for key in self.keys:
            data[key] = self.to_tensor(sample[key])

        data['img'] = data['img'].permute(2, 0, 1)
        return data

    def __repr__(self):
        return self.__class__.__name__ + f'(keys={self.keys})'


@TRANSFORMS.register_module()
class LoadImage:
    def __call__(self, results, keys=['img_path']):
        for key in keys:
            if isinstance(results[key], str):
                results[key.split('_')[0]] = cv2.imread(results[key])
            elif isinstance(results[key], torch.Tensor):
                results[key] = results[key].numpy()
            results[key].float()
        return results

@TRANSFORMS.register_module()
class ParseImage:
    def __call__(self, results):
        results['img_shape'] = results['img'].shape
        return results

@TRANSFORMS.register_module()
class ReOutDict:
    def __init__(self, **kwargs):
        self.remap = kwargs
    def __call__(self, results):
        new_result = dict()
        for key, value in self.remap.items():
            new_result[value] = results.pop(key)
        new_result['data_samples'] = results
        return new_result


@TRANSFORMS.register_module()
class Normalize(object):
    def __init__(self, img_norm):
        self.mean = np.array(img_norm['mean'], dtype=np.float32)
        self.std = np.array(img_norm['std'], dtype=np.float32)

    def __call__(self, sample):
        img = sample['img']
        if len(self.mean) == 1:
            img = img - np.array(self.mean)  # single channel image
            img = img / np.array(self.std)
        else:
            img = img - np.array(self.mean)[np.newaxis, np.newaxis, ...]
            img = img / np.array(self.std)[np.newaxis, np.newaxis, ...]
        sample['img'] = img

        return sample

    def imnormalize_(self, sample, mean, std, to_rgb=True):
        """Inplace normalize an image with mean and std.
        Args:
            img (ndarray): Image to be normalized.
            mean (ndarray): The mean to be used for normalize.
            std (ndarray): The std to be used for normalize.
            to_rgb (bool): Whether to convert to rgb.
        Returns:
            ndarray: The normalized image.
        """
        # cv2 inplace normalization does not accept uint8
        img = sample['img']
        assert img.dtype != np.uint8
        mean = np.float64(mean.reshape(1, -1))
        stdinv = 1 / np.float64(std.reshape(1, -1))
        if to_rgb:
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)  # inplace
        cv2.subtract(img, mean, img)  # inplace
        cv2.multiply(img, stdinv, img)  # inplace
        sample['img'] = img
        return sample