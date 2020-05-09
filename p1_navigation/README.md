[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"
[image2]: Documentation/scores_per_episode.png "Scores per episode"
[image3]: Documentation/scores_last_100.png "Last 100 episode scores"
# Project 1: Navigation

## Introduction

Project Navigation: In this project, we have to train an agent to navigate (and collect bananas!) in a large, square world. This environment is provided by [Unity Machine Learning agents] (https://github.com/Unity-Technologies/ml-agents).  

![Trained Agent][image1]

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of our agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, our agent must get an average score of +13 over 100 consecutive episodes.

## Getting Started

Download the environment from one of the links below.  You need only select the environment that matches our operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    

## Dependencies

1. Python 3.6
2. Unity mlagents
3. Unity unityagents
4. Tensorflow 1.7.1
5. Cudatoolkit 9.2
6. Pytorch 1.5

## Solution

I have implemented Deep Q networks using artificial neural networks and pixel input Deep Q networks using convolutional neural networks. I have also implemented experience replay and target networks to improve training.

1. Deep Q networks using artificial neural networks:
- [model.py](model.py) file contains the ANN model architecture
- [dqn_agent.py](dqn_agent.py) contains the Deep Q learning helper code.
- run the [Navigation.ipynb](Navigation.ipynb) file to train the model and visualize the agent.

### Results

![Scores per episode][image2]
![Last 100 episode scores][image3]


2. Deep Q networks using convolutional neural networks:
- [cnn_model.py](cnn_model.py) file contains the CNN model architecture
- [cnn_agent.py](cnn_agent.py) contains the Deep Q learning helper code.
- run the [Navigation_Pixels.ipynb](Navigation_Pixels.ipynb) file to train the model and visualize the agent.


