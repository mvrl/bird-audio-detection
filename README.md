# Entry for Bird Audio Detection Challenge

For more information about the challenge, see the [project
website](http://machine-listening.eecs.qmul.ac.uk/bird-audio-detection-challenge/)
or the [paper that summarizes the
challenge](https://arxiv.org/abs/1608.03417).

# Getting Started

This should be enough to get you training and evaluating code running:

## Clone the repository:

1. `git clone git@github.com:UkyVision/bird-audio-detection.git`
1. Create a branch for your work: `git checkout -b feature/{SHORT NAME OF FEATURE}`

## Install tensorflow:

1. Install [Miniconda](http://conda.pydata.org/miniconda.html).
1. Create a virtual environment called `tensorflow`: `conda create -n tensorflow python=2.7 ipython`.
1. Logout and log back in
1. Start the virtual environment: `source activate tensorflow`
1. Install Tensorflow using the [Anaconda provided
instructions](https://www.tensorflow.org/versions/master/get_started/os_setup.html#anaconda-installation). Short version: find
the correct setting for `TF_BINARY_URL` from the tensorflow website, then run
`pip install --ignore-installed --upgrade $TF_BINARY_URL`.

## Prepare the datasets:

1. Download the training datasets from the [project website](http://machine-listening.eecs.qmul.ac.uk/bird-audio-detection-challenge/)
1. Unpack the datasets (TODO: currently this goes in ~/data/birddetection, but we should probably put them in a standard place in the repo root, such as ./data/)
1. (currently not necessary) Create training/testing splits (TODO: write a script to make training and testing splits)

## Run the training script:

1. `cd ./src/`
1. `python main.py`: this will put checkpoints in the checkpoint directory
1. `python evaluate.py`: this uses the checkpoints to generate an output script 

## Training multiple models:

1. `cd ./src/driver`
1. `./driver.sh` : this uses GNU parallel to train multiple models.  it is currently configured to work on a machine
with two GPUs

# Contributing

If you are new to git and github, I encourage you to read this [guide to
contributing](http://blog.davidecoppola.com/2016/11/howto-contribute-to-open-source-project-on-github/).
Basically, read through the issues and/or talk to the team leaders to
see what would be a useful contribution. Then, you can either: 

- fork the repo (please keep it private for now), write your code,
    verify that it works, and submit a pull request into master when it is ready
- create a branch 'feature/*' and submit a pull request into master
    when you are ready

# Team

- Lead: Nathan Jacobs (jacobs@cs.uky.edu)

