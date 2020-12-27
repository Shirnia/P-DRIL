import os
import stable_baselines3

import numpy as np
import time

from stable_baselines3.common.cmd_util import make_vec_env
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import BaseCallback
from tqdm.auto import tqdm
from stable_baselines3.common import results_plotter


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """

    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf
        self.tic = time.perf_counter()


    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            toc = time.perf_counter()
            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), 'timesteps')
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
                if self.verbose > 0:
                    print("Num timesteps: {}".format(self.num_timesteps))
                    print(
                        "Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward,
                                                                                                 mean_reward))
                    print(f"time taken from start: {toc/60}mins")
                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    if self.verbose > 0:
                        print("Saving new best model to {}".format(self.save_path))
                    self.model.save(self.save_path)

        return True


class ProgressBarCallback(BaseCallback):
    def __init__(self, pbar):
        super(ProgressBarCallback, self).__init__()
        self._pbar = pbar
        self.tic = time.perf_counter()

    def _on_step(self):
        #toc = time.perf_counter()
        # Update the progress bar:
        self._pbar.n = self.num_timesteps
        self._pbar.update(0)
        #print(f"time passed: {self.tic-toc}")


class ProgressBarManager(object):
    def __init__(self, total_timesteps):  # init object with total timesteps
        self.pbar = None
        self.total_timesteps = total_timesteps

    def __enter__(self):  # create the progress bar and callback, return the callback
        self.pbar = tqdm(total=self.total_timesteps)

        return ProgressBarCallback(self.pbar)

    def __exit__(self, exc_type, exc_val, exc_tb):  # close the callback
        self.pbar.n = self.total_timesteps
        self.pbar.update(0)
        self.pbar.close()


# Create log dir
log_dir = "tmp/"
os.makedirs(log_dir, exist_ok=True)

# Create and wrap the environment
env = make_vec_env('foo-v0', n_envs=1, monitor_dir=log_dir)
tic = time.perf_counter()

# Create callbacks
save_callback = SaveOnBestTrainingRewardCallback(check_freq=10000, log_dir=log_dir)


model = stable_baselines3.DQN('MlpPolicy', env, verbose=0, learning_rate=1e-4)
model = model.load('AAA', env)


steps = 10e6
with ProgressBarManager(steps) as progress_callback:
    # This is equivalent to callback=CallbackList([progress_callback, auto_save_callback])
    model = model.learn(steps, callback=[progress_callback, save_callback])
model.save('AAA')
results_plotter.plot_results([log_dir], steps, results_plotter.X_TIMESTEPS, "TD3 LunarLander")