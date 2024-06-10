import numpy as np

class ImageTransforms(object):
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