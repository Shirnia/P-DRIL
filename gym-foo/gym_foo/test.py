import gym_foo.envs.foo_env as foo

a = foo.FooEnv()
isDone = False
while not isDone:
    b, c, d, e = a.step(0)
    a.render()
    isDone = d
