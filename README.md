# Camel Learning

A machine learning library written in OCaml with no external
dependencies. See `examples/` directory for example models.

This project is made purely for educational purposes and is not
suitable for use in performance demanding applications.

## Algorithms

Currently the two supported methods of learning are:

 - using [finite differences](https://en.wikipedia.org/wiki/Finite_difference),
 - using [back propagation](https://en.wikipedia.org/wiki/Backpropagation).

## Activation functions

Implemented activation functions to choose from are:

 - [Sigmoid](https://en.wikipedia.org/wiki/Sigmoid_function),
 - [ReLU](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)).

## MNIST database model

There is a working model trained on the
[MNIST](http://yann.lecun.com/exdb/mnist/) database. It trains the
fastest using sigmoid activation function and learning rate of 1.0,
but you can also use ReLU with learning rate of (from my tests) at
most 0.01. Using this configuration will be slower but you may achieve
better results, if you do, please share them with me :D
