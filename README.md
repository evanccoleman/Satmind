# Updates
Test commit

# Satmind
A reinforcement learning algorithm controller for a satellite using the Orekit library. The reinforcement learning algorithm 
is based on the Deep Deterministic Policy Gradient (DDPG) algorithm and prioritzed experience replay. The agent is a 
satellite that traverses a spacecraft enviornment. THe spacecraft's thruster is based on an electric proplution system which 
produces a low amount of thrust (< 1 N) with a long mission time (days).

A total of four missions (3 unique) were implemented: orbit raining, inclination change, semimajor axis change, and MEO to GEO orbit.

## Dependencies
Easiest way to install all the required packages is through Anaconda.

- Python 3.6 or later
- Tensorflow = 1.15
- Orekit >=10.0
- matplotlib for displaying results
- openai gym for testing RL algorithm

Use requirements.txt for easy setup using conda.

`conda create -f environment.yml`

## Usage

`python test_rl.py` 

tests to make sure RL algorithm is running correctly and runs in and openAI gym enviornment.

`python Satmind/orekit-env.py`

runs an orkit scenario that produces a contineous thrust, successfully configured if program does not crash.

### Pre-trained models

To run the pre-trained models, ensure that the input file points to the correct corresponding mission.

`python ddpg-sat.py --model <path to model>`

## Train from scratch

To run training from scratch pass:

`python ddpg-sat.py`  

## Arguments

optional arguments:

-  -h, --help         show this help message and exit
-  --model MODEL      path of a trained tensorlfow model (str: path)
-  --test             if testing a model (must pass a model as well)
-  --savefig          Save figures to file (saves in results directory)
-  --showfig          Display plotted figures

## Configure

- To change the orbit missions edit the input.json file. The initial and target states are in Keplarian orbital elements using degrees and the duration is expressed in days.
- To change the hyperparameters for the neural netowkrs or RL algorithm are in the ddpg-sat.py


# License

This software is distributed under the Apache 2.0 License.


