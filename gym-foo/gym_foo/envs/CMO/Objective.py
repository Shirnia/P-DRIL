import random as r
import numpy as np
import gym_foo.envs.CMO.Util as Util


class Objective:
    def __init__(self, israndom=False):
        self.iswon = False
        self.israndom = israndom
        self.pos_limits = np.int64([[40e3, 0, 0], [90e3, 40e3, 0]])
        self.radius = 5e3
        if israndom:
            self.position = np.float64([r.randint(self.pos_limits[0, 0], self.pos_limits[1, 0]),
                                        r.randint(self.pos_limits[0, 1], self.pos_limits[1, 1]),
                                        r.randint(self.pos_limits[0, 2], self.pos_limits[1, 2])])
        else:
            self.position = np.float64([50e3, 20e3, 0])

    def step(self, platform_pos):
        if Util.rss(self.position - platform_pos) < self.radius:
            self.iswon = True
