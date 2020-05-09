from agent import Agent
from monitor import interact
import gym
import numpy as np
import json

env = gym.make('Taxi-v3')
agent = Agent()
avg_rewards, best_avg_reward, best_Q_table = interact(env, agent)
json = json.dumps(best_Q_table)
f = open("best_Q_table.json","w")
f.write(json)
f.close()