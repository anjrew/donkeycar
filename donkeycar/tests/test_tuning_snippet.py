"""Tests for the web-tuning "paste myconfig snippet" path.

Covers the pure parser (`_parse_myconfig_snippet`) and the
`POST /tuning/snippet` handler, which applies a pasted snippet through the
normal validate/commit/broadcast pipeline.
"""
import json
import unittest

import tornado.web
from tornado import testing

from donkeycar.parts.web_controller.web import (
    LocalWebController,
    TuningSnippetHandler,
    _default_tuning,
    _MYCONFIG_TO_TUNING,
    _parse_myconfig_snippet,
    _render_myconfig_snippet,
)


class ParseSnippetTest(unittest.TestCase):
    """Pure-function tests for _parse_myconfig_snippet (no server)."""

    def test_assignment_lines(self):
        patch = _parse_myconfig_snippet(
            "LINE_FOLLOWER_MODE = 'center_line'\n"
            "PID_P = 0.5\n"
            "SCAN_Y = 12\n"
            "COLOR_THRESHOLD_LOW = (0, 50, 50)\n"
        )
        assert patch == {
            'line_follower_mode': 'center_line',
            'pid_p': 0.5,
            'scan_y': 12,
            'hsv_center_low': [0, 50, 50],
        }

    def test_commented_dict_entry_lines(self):
        # The PWM block is emitted as commented `#  "NAME": value,` lines.
        patch = _parse_myconfig_snippet(
            '#   "STEERING_LEFT_PWM":  290,\n'
            '#   "PWM_THROTTLE_SCALE": 1.5,\n'
        )
        assert patch == {'steering_left_pwm': 290, 'throttle_scale': 1.5}

    def test_inline_trailing_comments(self):
        # Regression: lines with an inline comment after the value used to be
        # silently dropped (the comment leaked into the captured value).
        patch = _parse_myconfig_snippet(
            "COLOR_THRESHOLD_LOW = (0, 50, 50)  # HSV dark yellow\n"
            "PID_P = 0.5  # proportional gain\n"
            'LINE_FOLLOWER_MODE = "edge_line"  # mode\n'
        )
        assert patch == {
            'hsv_center_low': [0, 50, 50],
            'pid_p': 0.5,
            'line_follower_mode': 'edge_line',
        }

    def test_unknown_keys_and_junk_ignored(self):
        patch = _parse_myconfig_snippet(
            "SOME_UNRELATED_CONST = 7\n"
            "# a free-form comment line\n"
            "\n"
            "not a config line at all\n"
            "PID_D = 0.25\n"
        )
        assert patch == {'pid_d': 0.25}

    def test_malformed_values_skipped(self):
        patch = _parse_myconfig_snippet(
            "PID_P = not_a_number\n"
            "COLOR_THRESHOLD_LOW = (0, 50)\n"   # wrong arity
            "SCAN_Y = 5\n"
        )
        assert patch == {'scan_y': 5}

    def test_round_trip_render_then_parse(self):
        # Everything _render_myconfig_snippet emits should parse back to the
        # same values. ai_throttle_mult is intentionally not rendered.
        tuning = _default_tuning()
        tuning.update({
            'line_follower_mode': 'center_line',
            'pid_p': 0.4, 'pid_i': 0.1, 'pid_d': 0.05,
            'throttle_min': 0.1, 'throttle_max': 0.8,
            'scan_y': 30, 'scan_height': 10,
            'hsv_center_low': [10, 20, 30],
            'steering_left_pwm': 290, 'steering_right_pwm': 490,
            'steering_scale': 1.2, 'throttle_scale': 0.9,
        })
        patch = _parse_myconfig_snippet(_render_myconfig_snippet(tuning))
        assert set(patch) == set(_MYCONFIG_TO_TUNING.values())
        for key in patch:
            assert patch[key] == tuning[key], key


class _TuningApp(tornado.web.Application):
    """Minimal app exposing just the tuning state and the snippet route."""
    apply_tuning_patch = LocalWebController.apply_tuning_patch
    broadcast_tuning = LocalWebController.broadcast_tuning

    def __init__(self):
        self.tuning = _default_tuning()
        self.tuning_seq = 0
        self.tuning_listeners = []
        self.wsTuningClients = []
        super().__init__([(r"/tuning/snippet", TuningSnippetHandler)])


class TuningSnippetHandlerTest(testing.AsyncHTTPTestCase):

    def get_app(self):
        self.app = _TuningApp()
        return self.app

    def test_post_applies_valid_snippet(self):
        body = "PID_P = 0.5\nSCAN_Y = 20\n"
        resp = self.fetch("/tuning/snippet", method="POST", body=body)
        assert resp.code == 200
        payload = json.loads(resp.body)
        assert set(payload["applied"]) == {"pid_p", "scan_y"}
        assert payload["rejections"] == []
        # State was actually committed.
        assert self.app.tuning["pid_p"] == 0.5
        assert self.app.tuning["scan_y"] == 20
        assert self.app.tuning_seq == 1

    def test_post_reports_rejections(self):
        # pid_p out of range (|v|>10) is rejected; scan_y still applies.
        body = "PID_P = 999\nSCAN_Y = 15\n"
        resp = self.fetch("/tuning/snippet", method="POST", body=body)
        payload = json.loads(resp.body)
        assert payload["applied"] == ["scan_y"]
        rejected = {r["key"] for r in payload["rejections"]}
        assert rejected == {"pid_p"}
        assert self.app.tuning["scan_y"] == 15

    def test_post_empty_snippet_is_noop(self):
        resp = self.fetch("/tuning/snippet", method="POST", body="")
        payload = json.loads(resp.body)
        assert payload["applied"] == []
        assert payload["rejections"] == []
        assert self.app.tuning_seq == 0
