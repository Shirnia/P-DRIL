import keyboard
import gym
import gym_foo

env = gym.make('foo-v0')
env.reset()
env.doPlot = True
isdone = False
prev_action = 0
t=0
while not isdone:
    action = 1
    hotkey = keyboard.get_hotkey_name()
    if hotkey != "":
        print(hotkey)
        if hotkey == "a":
            action = 2
        elif hotkey == "d":
            action = 0
        else:
            action = 1
    #if keyboard.is_pressed('a'):  # if key 'q' is pressed
    a,b,isdone,c = env.step(action)
    env.render()
    t+=env.ts
print(t)
