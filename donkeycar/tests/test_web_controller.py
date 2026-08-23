# -*- coding: utf-8 -*-
import pytest
import json
import os
from donkeycar.parts.web_controller.web import (
    LocalWebController,
    _parse_myconfig_snippet,
    _validate_tuning_patch,
)
import donkeycar.templates.cfg_complete as cfg
from importlib import reload


@pytest.fixture
def server():
    server = LocalWebController(cfg.WEB_CONTROL_PORT)
    return server


def test_json_output(server):
    result = server.run()
    json_result = json.dumps(result)
    d = json.loads(json_result)

    assert server.port == 8887

    assert d is not None
    assert int(d[0]) == 0


def test_web_control_user_defined_port():
    os.environ["WEB_CONTROL_PORT"] = "12345"
    reload(cfg)
    server = LocalWebController(port=cfg.WEB_CONTROL_PORT)

    assert server.port == 12345


def test_parse_myconfig_snippet_skips_commented_assignment_lines():
    """Commented-out plain assignments must NOT be applied (issue #99)."""
    snippet = "# STEERING_LEFT_PWM = 460\n# THROTTLE_MAX = 1.0\n# PID_P = 0.5\n"
    patch = _parse_myconfig_snippet(snippet)
    assert patch == {}, (
        "Commented-out myconfig assignment lines should be skipped, "
        f"but got: {patch}"
    )


def test_parse_myconfig_snippet_applies_commented_dict_entry_lines():
    """Commented dict-entry lines (PWM block format) must still be applied."""
    snippet = '  #   "STEERING_LEFT_PWM":  460,\n'
    patch = _parse_myconfig_snippet(snippet)
    assert (
        "steering_left_pwm" in patch
    ), f"Commented dict-entry lines should be parsed, but got: {patch}"
    assert patch["steering_left_pwm"] == 460


def test_parse_myconfig_snippet_applies_uncommented_assignment_lines():
    """Uncommented assignment lines must still be applied."""
    snippet = "PID_P = 0.7\nTHROTTLE_MAX = 0.9\n"
    patch = _parse_myconfig_snippet(snippet)
    assert patch.get("pid_p") == pytest.approx(0.7)
    assert patch.get("throttle_max") == pytest.approx(0.9)


_PWM_CURRENT = {
    "throttle_forward_pwm": 500,
    "throttle_stopped_pwm": 370,
    "throttle_reverse_pwm": 220,
}


def test_validate_tuning_patch_accepts_strict_pwm_ordering():
    clean, rejections = _validate_tuning_patch(
        {"throttle_forward_pwm": 520}, _PWM_CURRENT
    )
    assert clean.get("throttle_forward_pwm") == 520
    assert rejections == []


@pytest.mark.parametrize(
    "patch",
    [
        # forward == stopped: a full-throttle command emits the neutral pulse.
        {"throttle_forward_pwm": 370},
        # stopped == reverse: collapses the reverse range.
        {"throttle_reverse_pwm": 370},
        # inverted ordering.
        {"throttle_forward_pwm": 100},
    ],
)
def test_validate_tuning_patch_rejects_non_strict_pwm_ordering(patch):
    """Ordering must be strict: forward > stopped > reverse.

    Equality was previously permitted, which let a slider silently do
    nothing (RobotX-Workshops/donkey-cars#135, #136).
    """
    clean, rejections = _validate_tuning_patch(dict(patch), _PWM_CURRENT)
    key = next(iter(patch))
    assert key not in clean, f"{key} should have been rejected, got {clean}"
    assert any(r["key"] == key for r in rejections)
