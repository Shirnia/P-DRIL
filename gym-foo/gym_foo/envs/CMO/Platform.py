from typing import Final
import numpy as np
import gym_foo.envs.CMO.Util as Util


class Platform:
    def __init__(self, ts):
        self.isalive = True
        self.Ts: Final = ts
        self.prev_position = np.float64([1000, 10000, 0])
        self.position = np.float64([1000, 10000, 0])
        self.velocity = np.float64([Util.Mach, 0, 0])
        self.maxLatAx = 20
        self.fov = np.deg2rad(20)
        self.objective_in_fov = False

    def step(self, action, objective):
        self.prev_position = self.position.copy()
        # Rate of Turn
        # https://dspace.lib.cranfield.ac.uk/bitstream/handle/1826/2912/CHAPTER_8_july27_2.pdf?sequence=5&isAllowed=y
        ROT = np.float64(action) * 9.81 * np.sqrt(self.maxLatAx * self.maxLatAx - 1) / Util.rss(self.velocity)
        ROD = 0
        pol_vel = Util.cart2sph(self.velocity)
        pol_vel[0] = pol_vel[0] + ROT
        self.velocity = Util.sph2cart(pol_vel)
        # Apply accel to vel
        self.position += self.velocity * self.Ts
        self.objective_in_fov = Util.getAngle(self.velocity + self.position, self.position, objective.position) < self.fov / 2
