# NUS CS5446 Coursework Project AY25/26 semester 1
This repository is the course work projet for NUS CS5446 AI planning and decision making,
Project Team 27. 

Impelementation of RL environment and models built with gymnasium and stable-baselines3

Main files(in src):

mediator.py - game mecahnisms

rl_environment.py - reinfocement learning environment

train_dqn.py - training and evaluation of DQN model on the envrionment

train_ppo.py - training and evaluation of PPO model on the envrionment


# Installation
Install from requirements.txt
`pip install -r requirements.txt`


# Running RL
* run `src\train_dqn.py` or `src\train_ppo.py` with --mode "train" to train the respective algorithm. 
trained model will be saved in "models" folder

* run `src\train_dqn.py` or `src\train_ppo.py` with --mode "eval" to evaluate the trained models
supply --model paramter with the relative path to the model file
use --render parameter with "human" to render the game graphics for visualization

# python_mini_metro
The initial game code base was cloned from https://github.com/yanfengliu/python_mini_metro
This repo uses `pygame` to implement Mini Metro, a fun 2D strategic game where you try to optimize the max number of passengers your metro system can handle. Both human and program inputs are supported. One of the purposes of this implementation is to enable reinforcement learning agents to be trained on it.

## To play the game manually
* Run `src\main.py`


