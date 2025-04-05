import albumentations as A
import cv2

def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

def scale_down(x, **kwargs):
    return cv2.resize(x, (x.shape[0] // 8, x.shape[1] // 8))
    
def get_custom_transform():
    img_size = (480, 640)
    mean = [0.45, 0.45, 0.45]
    std = [0.25, 0.25, 0.25]
    additional_targets = { 'centroids': 'keypoints'}

    transform = A.Compose([
        A.ShiftScaleRotate(scale_limit=0.1, rotate_limit=0.1, shift_limit=0.1, p=0.1, border_mode=0),
        A.RandomCrop(height=img_size[0], width=img_size[1]),

        A.GaussNoise(p=0.2),
        A.Perspective(p=0.5),

        A.OneOf(
            [
                A.CLAHE(p=1),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1),
                A.RandomGamma(p=1),
            ],
            p=0.9,
        ),

        A.OneOf(
            [
                A.Sharpen(p=1),
                A.Blur(blur_limit=3, p=1),
                A.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),

        A.OneOf(
            [
                A.RandomBrightnessContrast(p=1),
                A.HueSaturationValue(p=1),
            ],
            p=0.9,
        )],
    additional_targets=additional_targets,    
    keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

    preprocessing_transform = A.Compose([
            A.Normalize(mean=mean, std=std),
            A.Lambda(mask=scale_down),
            A.Lambda(image=to_tensor, mask=to_tensor)],
        additional_targets=additional_targets,
        keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
    
    return transform, preprocessing_transform, mean, std