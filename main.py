# Main function of the RL air combat training programe
#
# Hasan ISCI - 27.12.2020

import gym
import gym_dubins_airplane
from matplotlib import pyplot as plt
import numpy as np

DebugInfo = False
RenderSteps = True

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # To test directly aircraft model
    # simple_ac = ACEnvironment2D()
    # simple_ac.takeaction(0.,0.,5.)

    env = gym.make('dubinsAC2D-v0', actions='discrete') # returns the environment
    state = env.reset()

    reward_history=[]

    # Run environment with arbitrary actions for testing purposes
    for _ in range(10000):

        state, reward, terminate, info = env.step(env.action_space.sample())  # take a random action

        reward_history.append(reward)

        if RenderSteps:
            env.render()
        if DebugInfo:
            print({'obs': state,
                   'reward': reward,
                   'terminate': terminate,
                   'info': info})
        if terminate:

            plt.plot( np.array(reward_history) )
            plt.show()

            reward_history.clear()

            state = env.reset()

    env.close()
