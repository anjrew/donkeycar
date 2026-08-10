import threading
import time

from .setup import on_pi

from donkeycar.parts.actuator import PCA9685, PWMSteering, PWMThrottle
import pytest


@pytest.mark.skipif(on_pi() == False, reason='Not on RPi')
def test_PCA9685():
    c = PCA9685(0)

@pytest.mark.skipif(on_pi() == False, reason='Not on RPi')
def test_PWMSteering():
    c = PCA9685(0)
    s = PWMSteering(c, 300, 440)


class MockPulseController:
    """ Records calls to set_pulse() without touching any hardware. """
    def __init__(self):
        self.calls = 0

    def set_pulse(self, pulse):
        self.calls += 1


def test_PWMSteering_update_does_not_busy_spin():
    """ update() runs on a background thread for as long as the vehicle is
    alive; it must not spin as fast as the interpreter allows, since that
    pins a CPU core and floods the PWM controller's I2C bus with redundant
    writes of an unchanged pulse. """
    controller = MockPulseController()
    steering = PWMSteering(controller, 300, 440)
    t = threading.Thread(target=steering.update)
    t.start()
    time.sleep(0.3)
    steering.shutdown()
    t.join(timeout=2)
    assert not t.is_alive()
    # rate-limited to ~100Hz; a busy spin would produce orders of magnitude
    # more calls in the same window
    assert controller.calls < 100


def test_PWMThrottle_update_does_not_busy_spin():
    controller = MockPulseController()
    throttle = PWMThrottle(controller, 400, 200, 300)
    calls_after_init = controller.calls
    t = threading.Thread(target=throttle.update)
    t.start()
    time.sleep(0.3)
    throttle.running = False
    t.join(timeout=2)
    assert not t.is_alive()
    assert controller.calls - calls_after_init < 100
