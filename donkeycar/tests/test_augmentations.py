"""
Unit-level sanity tests for the extended ImageAugmentation factory and the
label-aware horizontal mirror.
"""
import unittest

import numpy as np

from donkeycar.config import Config
from donkeycar.pipeline.augmentations import (
    ImageAugmentation, LABEL_AWARE_AUGMENTATIONS,
    CHANNEL_3_ONLY_AUGMENTATIONS)


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

    def test_hue_saturation(self):
        self._run_one('HUE_SATURATION', AUG_HUE_SHIFT_LIMIT=20,
                      AUG_SAT_SHIFT_LIMIT=30, AUG_VAL_SHIFT_LIMIT=20)

    def test_invert(self):
        self._run_one('INVERT')

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

    def _transform_class_names(self, aug):
        return {t.__class__.__name__ for t in aug.augmentations.transforms}

    def test_grayscale_skips_colour_only_augmentations(self):
        # IMAGE_DEPTH==1 must drop the colour-only augs (ISO_NOISE,
        # HUE_SATURATION) that need 3-channel images, while channel-agnostic
        # augs like BRIGHTNESS survive. Guards against the
        # "This transformation expects 3-channel images" crash.
        augs = ['ISO_NOISE', 'HUE_SATURATION', 'BRIGHTNESS']
        gray = ImageAugmentation(_cfg(AUGMENTATIONS=augs, IMAGE_DEPTH=1),
                                 'AUGMENTATIONS', prob=1.0, always_apply=True)
        names = self._transform_class_names(gray)
        self.assertNotIn('ISONoise', names)
        self.assertNotIn('HueSaturationValue', names)
        self.assertIn('RandomBrightnessContrast', names)

    def test_rgb_keeps_colour_only_augmentations(self):
        # IMAGE_DEPTH==3 (the default) must keep the colour-only augs.
        augs = ['ISO_NOISE', 'HUE_SATURATION', 'BRIGHTNESS']
        rgb = ImageAugmentation(_cfg(AUGMENTATIONS=augs, IMAGE_DEPTH=3),
                                'AUGMENTATIONS', prob=1.0, always_apply=True)
        names = self._transform_class_names(rgb)
        self.assertIn('ISONoise', names)
        self.assertIn('HueSaturationValue', names)
        self.assertIn('RandomBrightnessContrast', names)
        # The constants the skip logic relies on are what we think they are.
        self.assertEqual(CHANNEL_3_ONLY_AUGMENTATIONS,
                         {'ISO_NOISE', 'HUE_SATURATION'})


class _R:
    """Stand-in record. copy.copy() gives it a fresh identity, like a
    shallow-copied TubRecord."""
    def __init__(self, i=0):
        self.i = i


class TestMirrorDoubling(unittest.TestCase):
    """HORIZONTAL_FLIP is treated as synthetic data: _build_records emits every
    training record twice (original + mirror), doubling the set, and the x/y
    mirror decision must agree per record instance."""

    def _make_seq(self, enabled=True):
        # Import here to keep tensorflow off the import path for the
        # other tests.
        from donkeycar.pipeline.training import BatchSequence

        cfg = _cfg(AUG_HFLIP_SEED=42)
        # Bypass __init__ to skip TubSequence (it needs real records).
        seq = BatchSequence.__new__(BatchSequence)
        seq.config = cfg
        seq.is_train = True
        seq._mirror_enabled = enabled
        seq._mirror_decisions = {}
        seq._invert_enabled = False
        seq._invert_decisions = {}
        return seq

    def test_build_records_doubles_and_tags(self):
        seq = self._make_seq(enabled=True)
        records = [_R(0), _R(1), _R(2)]
        doubled = seq._build_records(records)

        # N -> 2N, all distinct objects.
        self.assertEqual(len(doubled), 6)
        self.assertEqual(len({id(r) for r in doubled}), 6)

        originals = [r for r in doubled if not seq._should_mirror(r)]
        twins = [r for r in doubled if seq._should_mirror(r)]
        self.assertEqual(len(originals), 3)
        self.assertEqual(len(twins), 3)
        # Each payload appears once as an original and once as a mirror twin.
        self.assertEqual(sorted(r.i for r in originals), [0, 1, 2])
        self.assertEqual(sorted(r.i for r in twins), [0, 1, 2])

    def test_should_mirror_agrees_per_instance(self):
        seq = self._make_seq(enabled=True)
        doubled = seq._build_records([_R(), _R()])
        for r in doubled:
            first = seq._should_mirror(r)
            for _ in range(5):
                self.assertEqual(seq._should_mirror(r), first)

    def test_disabled_passes_through_without_doubling(self):
        seq = self._make_seq(enabled=False)
        records = [_R(), _R()]
        out = seq._build_records(records)
        self.assertIs(out, records)
        for r in out:
            self.assertFalse(seq._should_mirror(r))

    def test_unregistered_record_is_never_mirrored(self):
        seq = self._make_seq(enabled=True)
        self.assertFalse(seq._should_mirror(_R()))


class TestInvertDoubling(unittest.TestCase):
    """INVERT is a photometric dataset-doubling pass (original + pixel-inverted
    twin) that does NOT negate the steering label. Combined with HORIZONTAL_FLIP
    it yields 4x data: original, flip, invert, flip+invert."""

    def _make_seq(self, mirror=False, invert=False):
        from donkeycar.pipeline.training import BatchSequence
        cfg = _cfg(AUG_HFLIP_SEED=42)
        seq = BatchSequence.__new__(BatchSequence)
        seq.config = cfg
        seq.is_train = True
        seq._mirror_enabled = mirror
        seq._mirror_decisions = {}
        seq._invert_enabled = invert
        seq._invert_decisions = {}
        return seq

    def test_invert_only_doubles_and_tags(self):
        seq = self._make_seq(invert=True)
        out = seq._build_records([_R(0), _R(1), _R(2)])
        # N -> 2N, all distinct objects.
        self.assertEqual(len(out), 6)
        self.assertEqual(len({id(r) for r in out}), 6)
        originals = [r for r in out if not seq._should_invert(r)]
        twins = [r for r in out if seq._should_invert(r)]
        self.assertEqual(len(originals), 3)
        self.assertEqual(len(twins), 3)
        self.assertEqual(sorted(r.i for r in originals), [0, 1, 2])
        self.assertEqual(sorted(r.i for r in twins), [0, 1, 2])
        # INVERT does not mirror, so no record should be flagged for mirror.
        self.assertFalse(any(seq._should_mirror(r) for r in out))

    def test_flip_and_invert_quadruples(self):
        seq = self._make_seq(mirror=True, invert=True)
        out = seq._build_records([_R(0), _R(1)])
        # N -> 4N.
        self.assertEqual(len(out), 8)
        self.assertEqual(len({id(r) for r in out}), 8)
        # All four (mirror, invert) combinations appear for each payload.
        combos = sorted((r.i, seq._should_mirror(r), seq._should_invert(r))
                        for r in out)
        expected = sorted(
            (i, m, v) for i in (0, 1)
            for m in (False, True) for v in (False, True))
        self.assertEqual(combos, expected)

    def test_invert_decision_stable_per_instance(self):
        seq = self._make_seq(invert=True)
        out = seq._build_records([_R(), _R()])
        for r in out:
            first = seq._should_invert(r)
            for _ in range(5):
                self.assertEqual(seq._should_invert(r), first)

    def test_disabled_passes_through_without_doubling(self):
        seq = self._make_seq(mirror=False, invert=False)
        records = [_R(), _R()]
        out = seq._build_records(records)
        self.assertIs(out, records)

    def test_unregistered_record_is_never_inverted(self):
        seq = self._make_seq(invert=True)
        self.assertFalse(seq._should_invert(_R()))


if __name__ == '__main__':
    unittest.main()
