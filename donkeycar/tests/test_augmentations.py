"""
Unit-level sanity tests for the extended ImageAugmentation factory and the
label-aware horizontal mirror.
"""
import unittest

import numpy as np

from donkeycar.config import Config
from donkeycar.pipeline.augmentations import (
    ImageAugmentation, LABEL_AWARE_AUGMENTATIONS)


def _cfg(**kwargs) -> Config:
    cfg = Config()
    for k, v in kwargs.items():
        setattr(cfg, k, v)
    return cfg


def _img(h=32, w=48, c=3) -> np.ndarray:
    rng = np.random.RandomState(0)
    return rng.randint(0, 255, size=(h, w, c), dtype=np.uint8)


class TestNewAugmentations(unittest.TestCase):
    """Each new aug type should produce a same-shape uint8 image."""

    def _run_one(self, aug_name: str, **extra):
        cfg = _cfg(AUGMENTATIONS=[aug_name], **extra)
        # always_apply=True forces the aug to fire regardless of prob.
        aug = ImageAugmentation(cfg, 'AUGMENTATIONS', prob=1.0, always_apply=True)
        out = aug.run(_img())
        self.assertEqual(out.shape, (32, 48, 3))
        self.assertEqual(out.dtype, np.uint8)

    def test_grayscale(self):
        self._run_one('GRAYSCALE')

    def test_cutout(self):
        self._run_one('CUTOUT', AUG_CUTOUT_MAX_HOLES=3, AUG_CUTOUT_MAX_SIZE=8,
                      AUG_CUTOUT_MIN_SIZE=4)

    def test_random_shadow(self):
        self._run_one('RANDOM_SHADOW')

    def test_iso_noise(self):
        self._run_one('ISO_NOISE')

    def test_bit_corruption(self):
        self._run_one('BIT_CORRUPTION', AUG_BIT_DROP_PROB=0.05)

    def test_jpeg_compression(self):
        self._run_one('JPEG_COMPRESSION', AUG_JPEG_QUALITY_RANGE=(40, 95))

    def test_motion_blur(self):
        self._run_one('MOTION_BLUR', AUG_MOTION_BLUR_LIMIT=5)

    def test_gauss_noise(self):
        self._run_one('GAUSS_NOISE', AUG_GAUSS_NOISE_VAR=(10.0, 30.0))

    def test_clahe(self):
        self._run_one('CLAHE')

    def test_posterize(self):
        self._run_one('POSTERIZE', AUG_POSTERIZE_BITS=4)

    def test_downscale(self):
        self._run_one('DOWNSCALE', AUG_DOWNSCALE_MIN=0.5, AUG_DOWNSCALE_MAX=0.9)

    def test_per_aug_prob_override(self):
        # AUG_<NAME>_PROB=0 should make the aug never fire even though the
        # global prob arg is 1.0.
        cfg = _cfg(AUGMENTATIONS=['BIT_CORRUPTION'], AUG_BIT_DROP_PROB=0.5,
                   AUG_BIT_CORRUPTION_PROB=0.0)
        aug = ImageAugmentation(cfg, 'AUGMENTATIONS', prob=1.0)
        img = _img()
        out = aug.run(img.copy())
        # With per-aug prob = 0, the image must be unchanged.
        np.testing.assert_array_equal(out, img)

    def test_brightness_still_works(self):
        # The existing BRIGHTNESS aug must still round-trip after our changes.
        self._run_one('BRIGHTNESS')

    def test_horizontal_flip_is_label_aware_marker(self):
        # HORIZONTAL_FLIP must NOT be included in the albumentations Compose:
        # the image flip happens in BatchSequence so it can also negate the
        # steering label.
        self.assertIn('HORIZONTAL_FLIP', LABEL_AWARE_AUGMENTATIONS)
        cfg = _cfg(AUGMENTATIONS=['HORIZONTAL_FLIP'])
        aug = ImageAugmentation(cfg, 'AUGMENTATIONS', prob=1.0, always_apply=True)
        # The Compose should be empty (no image-level transforms).
        self.assertEqual(len(aug.augmentations.transforms), 0)

    def test_unknown_aug_does_not_crash(self):
        cfg = _cfg(AUGMENTATIONS=['NOT_A_REAL_AUG_NAME'])
        aug = ImageAugmentation(cfg, 'AUGMENTATIONS')
        # Should produce an image-shaped output regardless.
        out = aug.run(_img())
        self.assertEqual(out.shape, (32, 48, 3))


class TestMirrorLabelFlip(unittest.TestCase):
    """The label-aware mirror in BatchSequence must agree between x and y."""

    def _make_seq(self, mirror_prob=1.0):
        # Import here to keep tensorflow off the import path for the
        # other tests.
        from donkeycar.pipeline.training import BatchSequence

        cfg = _cfg(
            AUGMENTATIONS=['HORIZONTAL_FLIP'],
            AUG_HFLIP_PROB=mirror_prob,
            AUG_HFLIP_SEED=42,
            TRANSFORMATIONS=[],
            POST_TRANSFORMATIONS=[],
            BATCH_SIZE=1,
        )
        # Bypass __init__ to skip TubSequence (it needs real records).
        seq = BatchSequence.__new__(BatchSequence)
        seq.config = cfg
        seq.is_train = True
        seq._mirror_enabled = True
        seq._mirror_prob = mirror_prob
        import random as _random
        seq._mirror_rng = _random.Random(42)
        return seq

    def test_mirror_decision_is_sticky(self):
        seq = self._make_seq(mirror_prob=0.5)

        class _R:
            pass

        r = _R()
        first = seq._should_mirror(r)
        # Repeated calls must return the same decision.
        for _ in range(10):
            self.assertEqual(seq._should_mirror(r), first)

    def test_mirror_always_when_prob_is_one(self):
        seq = self._make_seq(mirror_prob=1.0)

        class _R:
            pass

        for _ in range(20):
            self.assertTrue(seq._should_mirror(_R()))

    def test_mirror_never_when_prob_is_zero(self):
        seq = self._make_seq(mirror_prob=0.0)

        class _R:
            pass

        for _ in range(20):
            self.assertFalse(seq._should_mirror(_R()))

    def test_mirror_disabled_when_not_training(self):
        seq = self._make_seq(mirror_prob=1.0)
        seq._mirror_enabled = False

        class _R:
            pass

        self.assertFalse(seq._should_mirror(_R()))


if __name__ == '__main__':
    unittest.main()
