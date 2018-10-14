# DeepSat (SAT-6) Airborne Image Classification - Machine Learning 2 Final Project

## Overview
This repository contains the code and final paper for a project to use a convolutional neural network written in PyTorch to perform image classification on the Kaggle [DeepSat (SAT-6) Airborne](https://www.kaggle.com/crawford/deepsat-sat6) dataset. This was the final project for the Machine Learning 2 course at the George Washington University. It was a group project, however my contribution to the project was writing the code for the neural network, as well as explaining the theory behind the neural network (in the included paper).

## Dataset
The dataset is too large to host on GitHub. To download it, please follow these instructions:
1. run 'sudo pip install kaggle'
2. create a kaggle account
3. download your kaggle authentication json file
3. run 'mkdir Dataset' ##Create Dataset folder
5. run 'cd Dataset/'
4. run 'kaggle datasets download -d crawford/deepsat-sat6 -w' ##Download dataset to current directory
5. run 'ls' #should see all the files here
