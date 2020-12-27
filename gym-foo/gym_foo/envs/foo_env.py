import gym
import numpy as np
from typing import Final
from numpy.linalg import solve
from gym_foo.envs.CMO.Platform import Platform
from gym_foo.envs.CMO.Objective import Objective
from gym_foo.envs.CMO.Target import Target
import gym_foo.envs.CMO.Util as Util

from gym import spaces
import matplotlib.pyplot as plt
import matplotlib.lines as line


class FooEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.figure = plt.figure(figsize=(10, 8))
        self.isContinuous: Final = True
        self.max_time: Final = 300
        self.ts: Final = 1
        self.doPlot = False
        self.t = 0
        self.n_targets = 5
        self.max_targs_alive = 1
        self.platform = Platform(self.ts)
        self.targets = []
        self.objective = Objective(True)
        self.limits = np.float64([[0, 0, 0], [100e3, 40e3, 0]])
        self.action_curr = 0
        self.action_prev = 0
        self.reset()
        self._setup_plot()
        self.action_space = spaces.Box(low=-1, high=+1, shape=(1,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)
        self._action_map = {0: -1, 1: 0, 2: 1}
        self.reward_range = np.float64([0, 100])
        self.observation_space = spaces.Box(-1, +1, shape=self.get_state().shape, dtype=np.float32)

    def step(self, action_in):
        """

        Parameters
        ----------
        action :

        Returns
        -------
        ob, reward, episode_over, info : tuple
            ob (object) :
                an environment-specific object representing your observation of
                the environment.
            reward (float) :
                amount of reward achieved by the previous action. The scale
                varies between environments, but the goal is always to increase
                your total reward.
            episode_over (bool) :
                whether it's time to reset the environment again. Most (but not
                all) tasks are divided up into well-defined episodes, and done
                being True indicates the episode has terminated. (For example,
                perhaps the pole tipped too far, or you lost your last life.)
            info (dict) :
                 diagnostic information useful for debugging. It can sometimes
                 be useful for learning (for example, it might contain the raw
                 probabilities behind the environment's last state change).
                 However, official evaluations of your agent are not allowed to
                 use this for learning.
        """
        err_msg = "%r (%s) invalid" % (action_in, type(action_in))
        action = self._action_map[action_in]
        assert self.action_space.contains(action_in), err_msg
        self.action_curr = action
        self._take_action()
        self._step_environment()
        ob = self.get_state()
        episode_over = self._is_over()
        reward = self._get_reward()
        self.action_prev = action
        self.t += self.ts
        return ob, reward, episode_over, {}

    def reset(self):
        self.objective = Objective(True)
        self.platform = Platform(self.ts)
        self.targets = []
        for i in range(self.n_targets):
            self.targets.append(Target(self.ts, self.platform.position, self.platform.velocity))
            if i > self.max_targs_alive - 1:
                self.targets[i].isalive = False
        if self.figure.get_axes():
            self.figure.delaxes(self.figure.get_axes()[0])
        self.action_prev = 0
        self._setup_plot()
        self.t = 0
        return self.get_state()

    def render(self, mode='human', close=False):
        if self.doPlot:
            fig = self.figure
            ax0 = fig.axes[0]

            obj = self.platform
            ax0.collections[0].set_offsets([obj.position[0], obj.position[1]])
            # TODO - fix line2d
            lin = ax0.lines[0]
            lin.get_xdata().append(obj.position[0])
            lin.get_ydata().append(obj.position[1])

            for i in range(self.n_targets):
                obj = self.targets[i]
                ax0.collections[i+1].set_offsets([obj.position[0], obj.position[1]])
                # TODO - fix line2d
                lin = ax0.lines[i+1]
                lin.get_xdata().append(obj.position[0])
                lin.get_ydata().append(obj.position[1])

            plt.pause(0.00005)
            # plt.draw()

    def _step_environment(self):
        self.objective.step(self.platform.position)
        if not self.objective.iswon:
            for i in range(self.n_targets):
                self.targets[i].step(self.platform.position, self.platform.velocity)

    def _is_over(self):
        gate1 = self.objective.iswon
        gate2 = any(self.platform.position < self.limits[0, :])
        gate3 = any(self.platform.position > self.limits[1, :])
        gate4 = self.t > self.max_time
        gate5 = False
        for i in range(self.n_targets):
            if not gate5:
                gate5 = self._raycast(self.platform, self.targets[i])
        if any([gate1, gate2, gate3, gate4, gate5]):
            return True
        return False

    def _raycast(self, plat, targ):
        pA, Pb, dist = self._closestDistanceBetweenLines(plat.prev_position, plat.position,
                                                         targ.prev_position, targ.position, clampAll=True)
        return dist < targ.kill_range

    def _closestDistanceBetweenLines(self, a0, a1, b0, b1, clampAll=False, clampA0=False, clampA1=False, clampB0=False,
                                     clampB1=False):

        ''' Given two lines defined by numpy.array pairs (a0,a1,b0,b1)
            Return the closest points on each segment and their distance
            BS - No idea what clamps do
        '''

        # If clampAll=True, set all clamps to True
        if clampAll:
            clampA0 = True
            clampA1 = True
            clampB0 = True
            clampB1 = True

        # Calculate denomitator
        A = a1 - a0
        B = b1 - b0
        magA = np.linalg.norm(A)
        magB = np.linalg.norm(B)

        _A = A / magA
        _B = B / magB

        cross = np.cross(_A, _B);
        denom = np.linalg.norm(cross) ** 2

        # If lines are parallel (denom=0) test if lines overlap.
        # If they don't overlap then there is a closest point solution.
        # If they do overlap, there are infinite closest positions, but there is a closest distance
        if not denom:
            d0 = np.dot(_A, (b0 - a0))

            # Overlap only possible with clamping
            if clampA0 or clampA1 or clampB0 or clampB1:
                d1 = np.dot(_A, (b1 - a0))

                # Is segment B before A?
                if d0 <= 0 >= d1:
                    if clampA0 and clampB1:
                        if np.absolute(d0) < np.absolute(d1):
                            return a0, b0, np.linalg.norm(a0 - b0)
                        return a0, b1, np.linalg.norm(a0 - b1)


                # Is segment B after A?
                elif d0 >= magA <= d1:
                    if clampA1 and clampB0:
                        if np.absolute(d0) < np.absolute(d1):
                            return a1, b0, np.linalg.norm(a1 - b0)
                        return a1, b1, np.linalg.norm(a1 - b1)

            # Segments overlap, return distance between parallel segments
            return None, None, np.linalg.norm(((d0 * _A) + a0) - b0)

        # Lines criss-cross: Calculate the projected closest points
        t = (b0 - a0);
        detA = np.linalg.det([t, _B, cross])
        detB = np.linalg.det([t, _A, cross])

        t0 = detA / denom;
        t1 = detB / denom;

        pA = a0 + (_A * t0)  # Projected closest point on segment A
        pB = b0 + (_B * t1)  # Projected closest point on segment B

        # Clamp projections
        if clampA0 or clampA1 or clampB0 or clampB1:
            if clampA0 and t0 < 0:
                pA = a0
            elif clampA1 and t0 > magA:
                pA = a1

            if clampB0 and t1 < 0:
                pB = b0
            elif clampB1 and t1 > magB:
                pB = b1

            # Clamp projection A
            if (clampA0 and t0 < 0) or (clampA1 and t0 > magA):
                dot = np.dot(_B, (pA - b0))
                if clampB0 and dot < 0:
                    dot = 0
                elif clampB1 and dot > magB:
                    dot = magB
                pB = b0 + (_B * dot)

            # Clamp projection B
            if (clampB0 and t1 < 0) or (clampB1 and t1 > magB):
                dot = np.dot(_A, (pB - a0))
                if clampA0 and dot < 0:
                    dot = 0
                elif clampA1 and dot > magA:
                    dot = magA
                pA = a0 + (_A * dot)

        return pA, pB, np.linalg.norm(pA - pB)

    def _take_action(self):
        self.platform.step(self.action_curr, self.objective)

    def get_state(self):
        plat_pos = self.platform.position
        plat_vel = self.platform.velocity
        cat = np.concatenate([plat_pos, plat_vel], axis = None)
        for i in range(self.n_targets):
            targ_pos = self.targets[i].get_position()
            targ_vel = self.targets[i].get_velocity()
            cat = np.concatenate([cat, targ_pos, targ_vel], axis=None)
        cat = np.concatenate([cat, self.objective.position, self.action_prev], axis=None)
        scale = cat / Util.rss(self.limits[1, :] - self.limits[0, :])
        cat2 = np.concatenate([scale, self.action_prev], axis=None)
        # reshape = cat.reshape([10, 1, 1, 1])
        return cat2

    def _get_reward(self):
        """ Reward is given for XY. """
        logsize = 10000
        scale = Util.rss(self.limits[1, :])
        dpos = Util.rss(self.objective.position - self.platform.position)
        idx = np.int32((1-np.int32(dpos)/scale) * logsize)
        logrew_pos = np.logspace(-2, 2, logsize)
        rew = logrew_pos[idx]
        if self.platform.objective_in_fov:
            rew += 0.1
        if dpos < self.objective.radius:
            return 1000  # *(1-self.t/self.max_time)
        elif self._is_over() and Util.rss(self.platform.position - self.objective.position) > self.objective.radius:
            return -1000
        else:
            a = self.platform.position + self.platform.velocity
            b = self.platform.position
            c = self.objective.position
            return rew  # util.getAngle(a, b, c)/10
        # if self.action_curr == 0:
        #    return rew
        # else:
        #    return rew/10

    def set_plot(self, doPlot=True):
        self.doPlot = doPlot
        if doPlot:
            self._setup_plot()
            self.figure.show()

    def _setup_plot(self):
        fig = self.figure
        ax0 = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        ax0.set_xlim([self.limits[0, 0], self.limits[1, 0]])
        ax0.set_ylim([self.limits[0, 1], self.limits[1, 1]])

        obj = self.platform
        ax0.scatter(obj.position[0], obj.position[1], label="Platform")
        l0 = line.Line2D([obj.position[0]], [obj.position[1]], label="PlatformHistory")
        ax0.add_line(l0)

        obj = self.objective
        c0 = plt.Circle([obj.position[0], obj.position[1]], obj.radius, alpha=0.5, label="Objective")
        ax0.add_artist(c0)

        for i in range(self.n_targets):
            obj = self.targets[i]
            ax0.scatter(obj.position[0], obj.position[1], label=f"Target{i}")
            l1 = line.Line2D([obj.position[0]], [obj.position[1]], label="TargetHistory")
            ax0.add_line(l1)

        lims = [self.limits[0, 0], self.limits[1, 0], self.limits[0, 1], self.limits[1, 1]]
        ax0.axis(lims)
        ax0.set_aspect("equal")
        # fig.show()

    # def step(self, action):
    #    self.platform.step(action)
    #    self.target.step(self.platform.position, self.platform.velocity)
    #    if not self.platform.isalive and not self.target.isalive:
    #        isdone = True
    #    else:
    #        isdone = False
    #    return isdone
