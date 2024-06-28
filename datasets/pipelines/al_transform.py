import mmengine
import numpy as np
from .base_transform import BaseTransform
from mmengine import TRANSFORMS
import albumentations as AL

@TRANSFORMS.register_module()
class RandRotation(BaseTransform):
    def __init__(self, p=1, angle=(-90, 90), keys=['img']):
        super().__init__(p)
        self.angle = angle
        self.keys = keys
        self.p = p
    def transform(self, x: dict):
        for key in self.keys:
            x[key] = AL.Rotate(limit=self.angle, p=self.p)(image=x[key])['image']
        return x

@TRANSFORMS.register_module()
class RanddjustSharpness(BaseTransform):
    def __init__(self, p=1, alpha=(0.2, 0.5), keys=['img']):
        super().__init__(p)
        self.alpha = alpha
        self.keys = keys
        self.p = p
    def transform(self, x: dict):
        for key in self.keys:
            x[key] = AL.Sharpen(alpha=self.alpha, p=self.p)(image=x[key])['image']
        return x
@TRANSFORMS.register_module()
class RandColorJitter(BaseTransform):
    def __init__(self, p=1
                 , brightness = (0.8, 1), contrast = (0.8, 1), saturation = (0.8, 1), hue = (-0.5, 0.5)
                 , keys=['img']):
        super().__init__(p)
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.keys = keys
        self.p = p
    def transform(self, x: dict):
        for key in self.keys:
            x[key] = AL.ColorJitter(brightness=self.brightness, contrast=self.contrast, saturation=self.saturation, hue=self.hue, p=self.p)(image=x[key])['image']
        return x

@TRANSFORMS.register_module()
class RandErasing(BaseTransform):
    def __init__(self, p:float, max_holes=1, min_holes=0, max_width=8, min_width=0, min_height=0, max_height=0, keys=['img']):
        super().__init__(p)
        self.max_holes = max_holes
        self.min_holes = min_holes
        self.max_height = max_height
        if max_height == 0:
            self.max_height = max_width
        if min_height == 0:
            self.min_height = min_width
        
        self.max_width = max_width
        self.min_width = min_width    
        
        self.keys = keys
        self.p = p
    def transform(self, x: dict):
        for key in self.keys:
            x[key] = AL.CoarseDropout(max_holes=self.max_holes
                                      , min_holes=self.min_holes
                                      , max_height=self.max_height
                                      , max_width=self.max_width
                                      , min_height=self.min_height
                                      , min_width=self.min_width
                                      , fill_value=0
                                      , p=self.p)(image=x[key])['image']

        return x

class Flip(BaseTransform):
    def __init__(self, p:float, direction: str):
        super().__init__(p)
        self.direction = direction
    

    @staticmethod
    def imflip(img: np.ndarray, direction: str = 'horizontal') -> np.ndarray:
        """Flip an image horizontally or vertically.
        Args:
            img (ndarray): Image to be flipped.
            direction (str): The flip direction, either "horizontal" or
                "vertical" or "diagonal".
        Returns:
            ndarray: The flipped image.
        """
        assert direction in ['horizontal', 'vertical', 'diagonal']
        if direction == 'horizontal':
            return np.flip(img, axis=1)
        elif direction == 'vertical':
            return np.flip(img, axis=0)
        else:
            return np.flip(img, axis=(0, 1))

    def transform(self, results: dict) -> dict:
        img = results['img']
        results['img'] = self.imflip(img, direction=self.direction)
        return results