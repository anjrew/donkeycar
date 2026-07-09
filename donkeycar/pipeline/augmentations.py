import albumentations.core.transforms_interface
import logging
import albumentations as A


from donkeycar.config import Config


logger = logging.getLogger(__name__)


# HORIZONTAL_FLIP and INVERT are markers, not factory entries. They are
# handled in donkeycar.pipeline.training.BatchSequence as deterministic
# dataset-doubling passes (every record emitted twice: original + variant)
# rather than inside the albumentations Compose (which only sees the image).
# HORIZONTAL_FLIP also negates the steering label; INVERT does not.
LABEL_AWARE_AUGMENTATIONS = {'HORIZONTAL_FLIP', 'INVERT'}

# Augmentations that operate in colour space and require a 3-channel image.
# When training a grayscale model (IMAGE_DEPTH == 1) the tub loader feeds
# single-channel frames, so these are skipped: ISO_NOISE raises
# "This transformation expects 3-channel images", and HUE_SATURATION is a pure
# chroma op with nothing to act on. This lets the same AUGMENTATIONS list serve
# both RGB and grayscale models.
CHANNEL_3_ONLY_AUGMENTATIONS = {'ISO_NOISE', 'HUE_SATURATION'}


class ImageAugmentation:
    def __init__(self, cfg, key, prob=0.5, always_apply=False):
        skip = set(LABEL_AWARE_AUGMENTATIONS)
        if getattr(cfg, 'IMAGE_DEPTH', 3) == 1:
            skip |= CHANNEL_3_ONLY_AUGMENTATIONS
        requested = list(getattr(cfg, key, None) or [])
        aug_list = [a for a in requested if a not in skip]
        dropped = [a for a in requested
                   if a in CHANNEL_3_ONLY_AUGMENTATIONS and a in skip]
        if dropped:
            logger.info(f'IMAGE_DEPTH=1: skipping colour-only augmentations '
                        f'{dropped} (need 3-channel images)')
        # Each aug can override the global prob via AUG_<NAME>_PROB in config.
        # always_apply is no longer supported by albumentations 2.x; use p=1.0.
        augmentations = [
            ImageAugmentation.create(
                a, cfg,
                1.0 if always_apply else getattr(cfg, f'AUG_{a}_PROB', prob))
            for a in aug_list
        ]
        # Drop None entries from unknown keys so a typo doesn't crash Compose.
        self.augmentations = A.Compose([a for a in augmentations if a is not None])

    @classmethod
    def create(cls, aug_type: str, config: Config, prob: float) -> \
            albumentations.core.transforms_interface.BasicTransform:
        """ Augmentation factory — albumentations 2.x API.

        Cropping and trapezoidal mask are transformations which should be
        applied in training, validation and inference. Multiply, Blur and
        similar are augmentations which should be used only in training. """

        if aug_type == 'BRIGHTNESS':
            b_limit = getattr(config, 'AUG_BRIGHTNESS_RANGE', 0.2)
            c_limit = getattr(config, 'AUG_CONTRAST_RANGE', 0.5)
            logger.info(f'Creating augmentation {aug_type} brightness={b_limit} contrast={c_limit}')
            return A.RandomBrightnessContrast(
                brightness_limit=b_limit, contrast_limit=c_limit, p=prob)

        elif aug_type == 'BLUR':
            b_range = getattr(config, 'AUG_BLUR_RANGE', 3)
            logger.info(f'Creating augmentation {aug_type} {b_range}')
            return A.GaussianBlur(sigma_limit=b_range, blur_limit=(3, 25), p=prob)

        elif aug_type == 'GRAYSCALE':
            logger.info(f'Creating augmentation {aug_type}')
            return A.ToGray(p=prob)

        elif aug_type == 'CUTOUT':
            max_holes = getattr(config, 'AUG_CUTOUT_MAX_HOLES', 6)
            max_size  = getattr(config, 'AUG_CUTOUT_MAX_SIZE', 40)
            min_size  = getattr(config, 'AUG_CUTOUT_MIN_SIZE', 8)
            logger.info(
                f'Creating augmentation {aug_type} holes<=({1},{max_holes}) '
                f'size {min_size}..{max_size}')
            # albumentations 2.x: num_holes_range / hole_*_range (int = pixels)
            return A.CoarseDropout(
                num_holes_range=(1, max_holes),
                hole_height_range=(min_size, max_size),
                hole_width_range=(min_size, max_size),
                fill=0, p=prob)

        elif aug_type == 'RANDOM_SHADOW':
            num_lower  = getattr(config, 'AUG_SHADOW_NUM_LOWER', 1)
            num_upper  = getattr(config, 'AUG_SHADOW_NUM_UPPER', 2)
            intensity  = getattr(config, 'AUG_SHADOW_INTENSITY_RANGE', (0.3, 0.7))
            logger.info(
                f'Creating augmentation {aug_type} num {num_lower}..{num_upper} '
                f'intensity {intensity}')
            # albumentations 2.x: num_shadows_limit tuple, shadow_intensity_range
            return A.RandomShadow(
                num_shadows_limit=(num_lower, num_upper),
                shadow_intensity_range=intensity,
                p=prob)

        elif aug_type == 'ISO_NOISE':
            intensity = getattr(config, 'AUG_ISO_NOISE_RANGE', (0.01, 0.05))
            logger.info(f'Creating augmentation {aug_type} {intensity}')
            return A.ISONoise(intensity=intensity, p=prob)

        elif aug_type == 'BIT_CORRUPTION':
            # Sparse salt-and-pepper: drops individual pixels to random values.
            drop_prob = getattr(config, 'AUG_BIT_DROP_PROB', 0.01)
            logger.info(f'Creating augmentation {aug_type} drop={drop_prob}')
            return A.PixelDropout(
                dropout_prob=drop_prob, per_channel=True,
                drop_value=None,  # random per-pixel value
                p=prob)

        elif aug_type == 'JPEG_COMPRESSION':
            q_lo, q_hi = getattr(config, 'AUG_JPEG_QUALITY_RANGE', (40, 95))
            logger.info(f'Creating augmentation {aug_type} q={q_lo}..{q_hi}')
            # albumentations 2.x: quality_range tuple replaces quality_lower/upper
            return A.ImageCompression(quality_range=(q_lo, q_hi), p=prob)

        elif aug_type == 'MOTION_BLUR':
            blur_limit = getattr(config, 'AUG_MOTION_BLUR_LIMIT', 7)
            logger.info(f'Creating augmentation {aug_type} limit={blur_limit}')
            return A.MotionBlur(blur_limit=blur_limit, p=prob)

        elif aug_type == 'GAUSS_NOISE':
            # albumentations 2.x uses std_range (normalized 0-1) instead of
            # var_limit (pixel^2). Use AUG_GAUSS_NOISE_STD_RANGE if present;
            # fall back to converting the legacy AUG_GAUSS_NOISE_VAR.
            if hasattr(config, 'AUG_GAUSS_NOISE_STD_RANGE'):
                std_range = getattr(config, 'AUG_GAUSS_NOISE_STD_RANGE')
            else:
                var = getattr(config, 'AUG_GAUSS_NOISE_VAR', (10.0, 50.0))
                std_range = (var[0] ** 0.5 / 255.0, var[1] ** 0.5 / 255.0)
            logger.info(f'Creating augmentation {aug_type} std_range={std_range}')
            return A.GaussNoise(std_range=std_range, p=prob)

        elif aug_type == 'CLAHE':
            clip_limit = getattr(config, 'AUG_CLAHE_CLIP_LIMIT', 4.0)
            tile_grid  = getattr(config, 'AUG_CLAHE_TILE_GRID_SIZE', (8, 8))
            logger.info(
                f'Creating augmentation {aug_type} clip={clip_limit} '
                f'tile={tile_grid}')
            return A.CLAHE(clip_limit=clip_limit, tile_grid_size=tile_grid, p=prob)

        elif aug_type == 'POSTERIZE':
            num_bits = getattr(config, 'AUG_POSTERIZE_BITS', 4)
            logger.info(f'Creating augmentation {aug_type} bits={num_bits}')
            return A.Posterize(num_bits=num_bits, p=prob)

        elif aug_type == 'DOWNSCALE':
            # albumentations 2.x: scale_range tuple replaces scale_min/scale_max
            scale_min = getattr(config, 'AUG_DOWNSCALE_MIN', 0.5)
            scale_max = getattr(config, 'AUG_DOWNSCALE_MAX', 0.9)
            logger.info(
                f'Creating augmentation {aug_type} scale {scale_min}..{scale_max}')
            return A.Downscale(scale_range=(scale_min, scale_max), p=prob)

        elif aug_type == 'HUE_SATURATION':
            hue = getattr(config, 'AUG_HUE_SHIFT_LIMIT', 20)
            sat = getattr(config, 'AUG_SAT_SHIFT_LIMIT', 30)
            val = getattr(config, 'AUG_VAL_SHIFT_LIMIT', 20)
            logger.info(f'Creating augmentation {aug_type} h={hue} s={sat} v={val}')
            return A.HueSaturationValue(
                hue_shift_limit=hue, sat_shift_limit=sat, val_shift_limit=val,
                p=prob)

        elif aug_type == 'INVERT':
            # Marker only — handled as label-aware dataset doubling in training.py.
            # Should not appear here (skipped by LABEL_AWARE_AUGMENTATIONS filter).
            logger.warning(f'INVERT reached factory — should be label-aware only')
            return A.InvertImg(p=prob)

        logger.warning(f'Unknown augmentation type: {aug_type}')
        return None

    # Parts interface
    def run(self, img_arr):
        if len(self.augmentations) == 0:
            return img_arr
        aug_img_arr = self.augmentations(image=img_arr)["image"]
        return aug_img_arr
