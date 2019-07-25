#from maze_env import Maze
from model import DuelingDQN
import numpy as np
import gym

def train():
  step = 0
  for episode in range(1000):
    obs = env.reset()

    Reward = 0

    while True:
      # env.render()

      action = RL.choose_action(obs)

      ns, reward, done, _ = env.step(action)
      Reward += reward

      RL.store_transition(obs, action, reward, ns, done)

      if step > 200 and step % 5 == 0:
        RL.learn()

      obs = ns
      step += 1

      if done:
        print('episode: {}, Reward: {}'.format(episode, Reward))
        break

def _eval():
  for episode in range(10):
    obs = env.reset()

    Reward = 0

    while True:
      # env.render()

      action = RL.choose_action(obs, True)

      obs, reward, done, _ = env.step(action)
      Reward += reward

      if done:
        print('Reward: {}'.format(Reward))
        break
      
if __name__ == '__main__':
  env = gym.make('CartPole-v0')
  RL = DuelingDQN(env.observation_space.shape[0], env.action_space.n)

  train()

  _eval()
