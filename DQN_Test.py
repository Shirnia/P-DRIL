import os, time, stable_baselines3, gym

# Create log dir
log_dir = "tmp/"
os.makedirs(log_dir, exist_ok=True)
env_id = "foo-v0"

# Create and wrap the environment
env = env = gym.make(env_id)
env.doPlot = True
tic = time.perf_counter()

model = stable_baselines3.DQN('MlpPolicy', env, verbose=0, learning_rate=1e-4)
model = model.load('AAA', env)

observation = env.reset()
for t in range(100000):
        env.render()
        action, _states = model.predict(observation, deterministic=True)
        observation, reward, done, info = env.step(action)
        env.render()
        if done:
            print("Finished after {} timesteps".format(t+1))
            break