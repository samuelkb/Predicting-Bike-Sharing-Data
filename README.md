# Predicting Bike Sharing Data

## Introduction

In this project, I build a neural network from scratch to carry out a prediction problem on a real dataset. By building this neural network from the ground up, I improve my understanding of gradient descent, backpropagation, and other concepts that are important to know before to start working with PyTorch. And there is no better way than apply in a real world problem.

## The problem

Imagine yourself owning a bike sharing company. You want to predict how many bikes you need because if you have too few you are losing money from potential riders. If you have too many you are wasting money on bikes that are just sitting around. So you need to predict from historical data how many nikes you will need in the near future.

I do that with a neural network, which will be on the notebook.

## Running locally the project

In order to allow you to execute this project on your local machine, I provide a `requirements.txt` file. I highly recommed you to use a virtual environment, such as `virtenv` or `conda`.

The guide for conda:

1. Create a new conda environment with the following command: `conda create --name predicting-bikes python=3`
1. Enter your new enironment:
   1. For Mac / Linux: `conda activate predicting-bikes`
   1. For Windows: `activate predicting-bikes`
1. Ensure to have `numpy`, `matplotlib`, `pandas`, and `jupyter notebook` installed by doing the following: `conda install numpy matplotlib pandas jupyter notebook`

With that you have ready your environment for this project.
