from typing import Final
import numpy as np
import gym_foo.envs.CMO.Util as Util


class Target:
    def __init__(self, ts, target_pos=np.float64([0, 0, 0]), tarvet_vel=np.float64([0, 0, 0])):
        self.isalive = True
        self.isactive = False
        self.nav_type = "CLOS"
        self.N: Final = 5
        self.Ts: Final = ts
        self.kill_range: Final = 500 * self.Ts
        self.mach_no: Final = 3
        self.prev_position = np.float64([50e3, 0, 0])
        self.position = np.float64([50e3, 0, 0])
        self.velocity = np.float64([-Util.Mach * self.mach_no, 0, 0])
        self.target_position = target_pos
        self.target_velocity = tarvet_vel
        self.max_lat_ax = np.float64([40, 1, 1])
        self.start_range = 20e3

    def step(self, t_pos, t_vel):
        if self.isactive:
            temp_pos = self.position.copy()
            # use PN equation to get new vel/pos
            # https://en.wikipedia.org/wiki/Proportional_navigation
            if self.nav_type == "CLOS":
                self.CLOS(t_pos)
            elif self.nav_type == "PN":
                self.PN(t_pos, t_vel)
            self.prev_position = temp_pos
        else:
            if Util.rss(self.position - t_pos) < self.start_range:
                self.isactive = True

    def CLOS(self, t_pos):
        Pm = self.position
        Pt = t_pos
        Vm = self.velocity
        theta_t = Util.cart2sph(Pt - Pm)
        theta_m = Util.cart2sph(Vm)
        angle = Util.getAngle(Vm + Pm, Pm, Pt)

        # Max Rate of Turn
        # https://dspace.lib.cranfield.ac.uk/bitstream/handle/1826/2912/CHAPTER_8_july27_2.pdf?sequence=5&isAllowed=y
        ROT_max_rad = 9.81 * np.sqrt(self.max_lat_ax ** 2 - 1) / Util.rss(self.velocity) * self.Ts

        theta_m[0] += np.sign(theta_t[0] - theta_m[0]) * Util.absMin(ROT_max_rad[0], angle)

        self.velocity = Util.sph2cart(theta_m.copy())
        self.position += self.velocity * self.Ts

    def get_position(self):
        if self.isactive & self.isalive:
            return self.position
        else:
            return np.zeros((3, 1))

    def get_velocity(self):
        if self.isactive & self.isalive:
            return self.velocity
        else:
            return np.zeros((3, 1))

    def PN(self, t_pos, t_vel):
        # TODO - FIX PN

        # To find omega:
        #   omega = (R*Vr)/dot(R,R)
        #   Vr = Vt - Vm
        #   Rr = Rt - Rm
        Vm = self.velocity
        Rm = self.position
        Rr = t_pos - Rm
        Vr = t_vel - Vm
        eq1 = np.cross(Rr, Vr)
        eq2 = np.dot(Rr, Rr)
        omega = np.divide(eq1, eq2)

        # To find accel:
        #   a = -N*|Vr|*(Vm/|Vm|)*omega
        #   eq1 = (Vm/|Vm|)
        #   eq2 = N*|Vr|
        #   eq3 = cross(eq1*eq2,omega)
        n = self.N
        eq1 = np.divide(Vm, Util.rss(Vm))
        a = n * np.cross(eq1, omega)
        a_sph_rad = Util.cart2sph(a)
        v_sph_rad = a_sph_rad * self.Ts
        tmp_velocity_sph = Util.cart2sph(self.velocity.copy())

        # Max Rate of Turn
        # https://dspace.lib.cranfield.ac.uk/bitstream/handle/1826/2912/CHAPTER_8_july27_2.pdf?sequence=5&isAllowed=y
        ROT_max_rad = 9.81 * np.sqrt(self.max_lat_ax ** 2 - 1) / Util.rss(self.velocity) * self.Ts
        ROT_max_deg = np.rad2deg(ROT_max_rad)

        # if a_sph_deg < ROT_max_deg:
        # min(v_sph_rad[0], ROT_max_rad[0])
        tmp_velocity_sph[0] += v_sph_rad[0]
        tmp_velocity = Util.sph2cart(tmp_velocity_sph)
        # else:
        #     tmp_velocity += ROT_max_deg * self.Ts
        self.velocity = tmp_velocity.copy()
        self.position += np.sign(a) * self.velocity * self.Ts
