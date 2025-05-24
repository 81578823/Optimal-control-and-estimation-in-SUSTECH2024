import os
import math
import matplotlib.pyplot as plt
import torch

from network import DDPGAgent
from utils import *
def evaluate():

    # simulation of the agent solving the invented pendulum swing-up problem

    env = gym.make('InvertedPendulumCustom-v0',render_mode="human", width=320, height=240) 

    curr_dir = os.path.abspath(os.getcwd())
    agent = torch.load(curr_dir + "/models/pendulum_swingup_ddpg_cpu.pkl", map_location=torch.device('cpu'))
    agent.train = False

    state,info = env.reset()
    r = 0
    theta = []
    actions = []
    for i in range(500):
        action = agent.get_action(state)
        next_state, reward, done, _ , _ = env.step(action)     # render() is in this def
        actions.append(action)
        theta.append(math.degrees(next_state[2]))
        r += reward
        state = next_state

    env.close()
    # plot the angle and action curve
    curr_dir = os.path.abspath(os.getcwd())
    if not os.path.isdir("results"):
        os.mkdir("results")

    plt.figure()
    plt.plot(theta)
    plt.title('Angle')
    plt.ylabel('Angle in degrees')
    plt.xlabel('Time step t')
    plt.savefig(curr_dir + "/results/plot_angle.png")

    plt.figure()
    plt.plot(actions)
    plt.title('Action')
    plt.ylabel('Action in Newton')
    plt.xlabel('Time step t')
    plt.savefig(curr_dir + "/results/plot_action.png")


if __name__ == '__main__':
    evaluate()