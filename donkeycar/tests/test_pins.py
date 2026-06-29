import sys
import types

import pytest

from donkeycar.parts.pins import PCA9685


def _install_fake_adafruit(monkeypatch, raise_errno=None, default_bus=None):
    """Install fake Adafruit_PCA9685 / Adafruit_GPIO modules.

    If raise_errno is set, the fake PCA9685 constructor raises an OSError
    with that errno so the error-handling branch under test runs without
    real hardware.
    """
    pca_mod = types.ModuleType("Adafruit_PCA9685")

    class _FakePCA9685:
        def __init__(self, address=0x40):
            if raise_errno is not None:
                raise OSError(raise_errno, "Remote I/O error")

        def set_pwm_freq(self, freq):
            pass

    pca_mod.PCA9685 = _FakePCA9685
    monkeypatch.setitem(sys.modules, "Adafruit_PCA9685", pca_mod)

    gpio_pkg = types.ModuleType("Adafruit_GPIO")
    i2c_mod = types.ModuleType("Adafruit_GPIO.I2C")
    i2c_mod.get_default_bus = lambda: default_bus
    gpio_pkg.I2C = i2c_mod
    monkeypatch.setitem(sys.modules, "Adafruit_GPIO", gpio_pkg)
    monkeypatch.setitem(sys.modules, "Adafruit_GPIO.I2C", i2c_mod)


def test_pca9685_init_success(monkeypatch):
    _install_fake_adafruit(monkeypatch)
    pca = PCA9685(busnum=1, address=0x40, frequency=60)
    assert pca.get_frequency() == 60


def test_pca9685_init_oserror_has_helpful_message(monkeypatch):
    _install_fake_adafruit(monkeypatch, raise_errno=121)
    with pytest.raises(OSError) as exc_info:
        PCA9685(busnum=1, address=0x40, frequency=60)
    err = exc_info.value
    # Original exception is chained, so its errno stays accessible.
    assert isinstance(err.__cause__, OSError)
    assert err.__cause__.errno == 121
    msg = str(err)
    assert "PCA9685" in msg
    assert "0x40" in msg
    assert "i2cdetect -y 1" in msg


def test_pca9685_init_oserror_uses_default_bus_when_busnum_none(monkeypatch):
    # busnum None -> hint comes from get_default_bus(), not a hardcoded 1.
    _install_fake_adafruit(monkeypatch, raise_errno=121, default_bus=7)
    with pytest.raises(OSError) as exc_info:
        PCA9685(busnum=None, address=0x40, frequency=60)
    assert "i2cdetect -y 7" in str(exc_info.value)
