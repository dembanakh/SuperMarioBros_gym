import gym.wrappers as wr


def wrapper(env):
    env = wr.gray_scale_observation.GrayScaleObservation(env)
    env = wr.resize_observation.ResizeObservation(env, 84)
    env = wr.frame_stack.FrameStack(env, 4)
    return env
