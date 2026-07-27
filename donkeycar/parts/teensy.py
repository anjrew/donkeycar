from datetime import datetime
import donkeycar as dk
import re
import time

class TeensyRCin:
    def __init__(self):
        self.inSteering = 0.0
        self.inThrottle = 0.0

        self.sensor = dk.parts.actuator.Teensy(0)

        TeensyRCin.LEFT_ANGLE = -1.0
        TeensyRCin.RIGHT_ANGLE = 1.0
        TeensyRCin.MIN_THROTTLE = -1.0
        TeensyRCin.MAX_THROTTLE =  1.0

        TeensyRCin.LEFT_PULSE = 496.0
        TeensyRCin.RIGHT_PULSE = 242.0
        TeensyRCin.MAX_PULSE = 496.0
        TeensyRCin.MIN_PULSE = 242.0


        self.on = True

    def update(self):
        rcin_pattern = re.compile('^I +([.0-9]+) +([.0-9]+).*$')

        while self.on:
            start = datetime.now()

            l = self.sensor.teensy_readline()

            while l:
                # print("mw TeensyRCin line= " + l.decode('utf-8'))
                m = rcin_pattern.match(l.decode('utf-8'))

                if m:
                    i = float(m.group(1))
                    if i == 0.0:
                        self.inSteering = 0.0
                    else:
                        i = i / (1000.0 * 1000.0) # in seconds
                        i *= self.sensor.frequency * 4096.0
                        self.inSteering = dk.utils.map_frange(i,
                                                         TeensyRCin.LEFT_PULSE, TeensyRCin.RIGHT_PULSE,
                                                         TeensyRCin.LEFT_ANGLE, TeensyRCin.RIGHT_ANGLE)

                    k = float(m.group(2))
                    if k == 0.0:
                        self.inThrottle = 0.0
                    else:
                        k = k / (1000.0 * 1000.0) # in seconds
                        k *= self.sensor.frequency * 4096.0
                        self.inThrottle = dk.utils.map_frange(k,
                                                         TeensyRCin.MIN_PULSE, TeensyRCin.MAX_PULSE,
                                                         TeensyRCin.MIN_THROTTLE, TeensyRCin.MAX_THROTTLE)

                    # print("matched %.1f  %.1f  %.1f  %.1f" % (i, self.inSteering, k, self.inThrottle))
                l = self.sensor.teensy_readline()

            stop = datetime.now()
            s = 0.01 - (stop - start).total_seconds()
            if s > 0:
                time.sleep(s)

    def run_threaded(self):
        return self.inSteering, self.inThrottle

    def shutdown(self):
        # indicate that the thread should be stopped
        self.on = False
        print('stopping TeensyRCin')
        time.sleep(.5)

