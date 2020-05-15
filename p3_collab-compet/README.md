[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif "Trained Agent"
[image2]: Documentation/ddpg_structure.png "DDPG architecture"
[image3]: Documentation/scores_per_episode.png "Scores per episode"


# Project 3: Collaboration and Competition

## Introduction

For this project, you will work with the [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment.

![Trained Agent][image1]

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation.  Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. 

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single **score** for each episode.

The environment is considered solved, when the average (over 100 episodes) of those **scores** is at least +0.5.

## Getting Started

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux_NoVis.zip) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (_To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)

2. Place the file in the DRLND GitHub repository, in the `p3_collab-compet/` folder, and unzip (or decompress) the file. 

## Dependencies

1. Python 3.6
2. Unity mlagents
3. Unity unityagents
4. Tensorflow 1.7.1
5. Cudatoolkit 9.2
6. Pytorch 1.5  

## Solution

- [model.py](model.py) file contains the Actor-Critic model architecture.
- [DDPGAgents.py](DDPGAgents.py) Create an DDPGAgents class that interacts with and learns from the environment.
- [ReplayBuffer.py](ReplayBuffer.py) Replay Buffer class to store the experiences.
- [OUNoise.py](OUNoise.py) Ornstein Uhlenbeck noise for the actor to improve exploration.
- [config.json](config.json) Configuration file to store variables and paths.
- [utils.py](utils.py) Helper functions.    
- run the [Tennis.ipynb](Tennis.ipynb) file to train the model and visualize the agent.

![DDPG Architecture][image2]

There is a modification to the DDPG agent used in the 'continuous control - robot arm' project. The main difference is that here, all the agents share the same replay buffer memory. The main reason that the agent may get caught in a tight loop of specific states that loop back through one another and detract from the agent exploring the full environment "equally". By sharing a common replay buffer, we ensure that we explore the whole environment. Both the agents still have their own actor critic networks to train. 

Because we have two models for each agent; the actor and critic that must be trained, it means that we have two set of weights that must be optimized separately. Adam was used for the neural networks with a learning rate of 10−4 and 10−3 respectively for the actor and critic for each agent. * For Q, I used a discount factor of γ = 0.99. For the soft target updates, the hyperparameter τ was set to 0.001. The neural networks have 2 hidden layers with 250 and 100 units respectively. For the critic Q, the actions were not included the 1st hidden layer of Q. The ﬁnal layer weights and biases of both the actor and critic were initialized from a uniform distribution [−3×10<sup>−3</sup>,3×10<sup>−3</sup>] and [3×10<sup>−4</sup>;3×10<sup>−4</sup>] to ensure that the initial outputs for the policy and value estimates were near zero. As for the layers, they were initialized from uniform distribution [− 1 / √f , 1 / √f ] where f is the fan-in of the layer. We use the prelu1 activation function for each layer.

The following is the graph of the scores per episode as the models train:

![Scores per episode][image3]

The environment is solved in 1809 episodes.

