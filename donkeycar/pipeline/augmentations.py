import albumentations.core.transforms_interface
import logging
import albumentations as A
from albumentations import GaussianBlur
from albumentations.augmentations.transforms import RandomBrightnessContrast

from donkeycar.config import Config


logger = logging.getLogger(__name__)


# HORIZONTAL_FLIP is a marker, not a factory entry. The image flip and the
# steering-label negation must happen together, so it is handled in
# donkeycar.pipeline.training.BatchSequence rather than inside the
# albumentations Compose (which only sees the image).
LABEL_AWARE_AUGMENTATIONS = {'HORIZONTAL_FLIP'}


class ImageAugmentation:
    def __init__(self, cfg, key, prob=0.5, always_apply=False):
        aug_list = [a for a in getattr(cfg, key, [])
                    if a not in LABEL_AWARE_AUGMENTATIONS]
        # Each aug can override the global prob via AUG_<NAME>_PROB in config.
        augmentations = [
            ImageAugmentation.create(
                a, cfg,
                getattr(cfg, f'AUG_{a}_PROB', prob),
                always_apply)
            for a in aug_list
        ]
        # Drop None entries from unknown keys so a typo doesn't crash Compose.
        self.augmentations = A.Compose([a for a in augmentations if a is not None])

    @classmethod
    def create(cls, aug_type: str, config: Config, prob, always) -> \
            albumentations.core.transforms_interface.BasicTransform:
        """ Augmenatition factory. Cropping and trapezoidal mask are
            transfomations which should be applied in training, validation
            and inference. Multiply, Blur and similar are augmentations
            which should be used only in training. """

        if aug_type == 'BRIGHTNESS':
            b_limit = getattr(config, 'AUG_BRIGHTNESS_RANGE', 0.2)
            logger.info(f'Creating augmentation {aug_type} {b_limit}')
            return RandomBrightnessContrast(brightness_limit=b_limit,
                                            contrast_limit=b_limit,
                                            p=prob, always_apply=always)

        elif aug_type == 'BLUR':
            b_range = getattr(config, 'AUG_BLUR_RANGE', 3)
            logger.info(f'Creating augmentation {aug_type} {b_range}')
            return GaussianBlur(sigma_limit=b_range, blur_limit=(13, 13),
                                p=prob, always_apply=always)

        elif aug_type == 'GRAYSCALE':
            # Replaces RGB with chroma-stripped RGB (still 3 channels).
            # If IMAGE_DEPTH == 1, the POST_TRANSFORMATIONS 'RGB2GRAY' step
            # collapses to a single channel after augmentation.
            logger.info(f'Creating augmentation {aug_type}')
            return A.ToGray(p=prob, always_apply=always)

        elif aug_type == 'CUTOUT':
            max_holes = getattr(config, 'AUG_CUTOUT_MAX_HOLES', 6)
            max_size = getattr(config, 'AUG_CUTOUT_MAX_SIZE', 40)
            min_size = getattr(config, 'AUG_CUTOUT_MIN_SIZE', 8)
            logger.info(
                f'Creating augmentation {aug_type} holes<={max_holes} '
                f'size {min_size}..{max_size}')
            return A.CoarseDropout(
                max_holes=max_holes, max_height=max_size, max_width=max_size,
                min_holes=1, min_height=min_size, min_width=min_size,
                fill_value=0, p=prob, always_apply=always)

        elif aug_type == 'RANDOM_SHADOW':
            num_lower = getattr(config, 'AUG_SHADOW_NUM_LOWER', 1)
            num_upper = getattr(config, 'AUG_SHADOW_NUM_UPPER', 2)
            logger.info(
                f'Creating augmentation {aug_type} num {num_lower}..{num_upper}')
            return A.RandomShadow(
                num_shadows_lower=num_lower, num_shadows_upper=num_upper,
                p=prob, always_apply=always)

        elif aug_type == 'ISO_NOISE':
            intensity = getattr(config, 'AUG_ISO_NOISE_RANGE', (0.01, 0.05))
            logger.info(f'Creating augmentation {aug_type} {intensity}')
            return A.ISONoise(intensity=intensity, p=prob, always_apply=always)

        elif aug_type == 'BIT_CORRUPTION':
            # Sparse salt-and-pepper: drops individual pixels to random values.
            drop_prob = getattr(config, 'AUG_BIT_DROP_PROB', 0.01)
            logger.info(f'Creating augmentation {aug_type} drop={drop_prob}')
            return A.PixelDropout(
                dropout_prob=drop_prob, per_channel=True,
                drop_value=None,  # random per-pixel value
                p=prob, always_apply=always)

        elif aug_type == 'JPEG_COMPRESSION':
            q_lo, q_hi = getattr(config, 'AUG_JPEG_QUALITY_RANGE', (40, 95))
            logger.info(f'Creating augmentation {aug_type} q={q_lo}..{q_hi}')
            return A.ImageCompression(
                quality_lower=q_lo, quality_upper=q_hi,
                p=prob, always_apply=always)

        elif aug_type == 'MOTION_BLUR':
            # Directional blur — simulates fast motion / camera shake.
            blur_limit = getattr(config, 'AUG_MOTION_BLUR_LIMIT', 7)
            logger.info(f'Creating augmentation {aug_type} limit={blur_limit}')
            return A.MotionBlur(blur_limit=blur_limit, p=prob,
                                always_apply=always)

        elif aug_type == 'GAUSS_NOISE':
            var_limit = getattr(config, 'AUG_GAUSS_NOISE_VAR', (10.0, 50.0))
            logger.info(f'Creating augmentation {aug_type} var={var_limit}')
            return A.GaussNoise(var_limit=var_limit, p=prob,
                                always_apply=always)

        elif aug_type == 'CLAHE':
            # Contrast-limited adaptive histogram eq. Useful pre-grayscale to
            # equalise lighting across the frame.
            clip_limit = getattr(config, 'AUG_CLAHE_CLIP_LIMIT', 4.0)
            tile_grid = getattr(config, 'AUG_CLAHE_TILE_GRID_SIZE', (8, 8))
            logger.info(
                f'Creating augmentation {aug_type} clip={clip_limit} '
                f'tile={tile_grid}')
            return A.CLAHE(clip_limit=clip_limit, tile_grid_size=tile_grid,
                           p=prob, always_apply=always)

        elif aug_type == 'POSTERIZE':
            num_bits = getattr(config, 'AUG_POSTERIZE_BITS', 4)
            logger.info(f'Creating augmentation {aug_type} bits={num_bits}')
            return A.Posterize(num_bits=num_bits, p=prob,
                               always_apply=always)

        elif aug_type == 'DOWNSCALE':
            # Downsample then upsample — simulates a worse camera / lossy pipe.
            scale_min = getattr(config, 'AUG_DOWNSCALE_MIN', 0.5)
            scale_max = getattr(config, 'AUG_DOWNSCALE_MAX', 0.9)
            logger.info(
                f'Creating augmentation {aug_type} scale {scale_min}..{scale_max}')
            return A.Downscale(scale_min=scale_min, scale_max=scale_max,
                               p=prob, always_apply=always)

        logger.warning(f'Unknown augmentation type: {aug_type}')
        return None

    # Parts interface
    def run(self, img_arr):
        aug_img_arr = self.augmentations(image=img_arr)["image"]
        return aug_img_arr

