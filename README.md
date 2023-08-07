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

There is an attempt at creating a MNIST database model in the examples
directory but I could not get any decent results from it. Two possible
causes I can think of are: the use of sigmoid activation function
instead of ReLU.  And the fact that this library is too slow: the
actual training must probably be done with a much lower learning rate
but decreasing it also means waiting longer for the results.
