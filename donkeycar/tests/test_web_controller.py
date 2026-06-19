# -*- coding: utf-8 -*-
import pytest
import json
import os
from donkeycar.parts.web_controller.web import (
    LocalWebController,
    _parse_myconfig_snippet,
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
    snippet = "# STEERING_LEFT_PWM = 460\n" "# THROTTLE_MAX = 1.0\n" "# PID_P = 0.5\n"
    patch = _parse_myconfig_snippet(snippet)
    assert patch == {}, (
        "Commented-out myconfig assignment lines should be skipped, "
        f"but got: {patch}"
    )


def test_parse_myconfig_snippet_applies_commented_dict_entry_lines():
    """Commented dict-entry lines (PWM block format) must still be applied."""
    snippet = '  #   "STEERING_LEFT_PWM":  460,\n'
    patch = _parse_myconfig_snippet(snippet)
    assert "steering_left_pwm" in patch, (
        "Commented dict-entry lines should be parsed, " f"but got: {patch}"
    )
    assert patch["steering_left_pwm"] == 460


def test_parse_myconfig_snippet_applies_uncommented_assignment_lines():
    """Uncommented assignment lines must still be applied."""
    snippet = "PID_P = 0.7\nTHROTTLE_MAX = 0.9\n"
    patch = _parse_myconfig_snippet(snippet)
    assert patch.get("pid_p") == pytest.approx(0.7)
    assert patch.get("throttle_max") == pytest.approx(0.9)
