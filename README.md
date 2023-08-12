# Camel Learning

A machine learning library written in OCaml with no external
dependencies. See `examples/` directory for example models.

This project is made purely for educational purposes and is not
suitable for use in performance demanding applications.

## MNIST model visualization

![Screenshot of running `examples/mnist_sdl.ml`](images/mnist_sdl.png)

There is a working model trained on the
[MNIST](http://yann.lecun.com/exdb/mnist/) database.

To run the SDL demo you first need to train the model using
`examples/mnist_train.ml`.

By default it trains using sigmoid activation function and learning
rate of 1.0, this is what I found to be the best configuration, but
you can also use ReLU with learning rate of (from my tests) at most
0.01. Using this configuration will be slower but you may achieve
better results.

Interrupting the program with `C-c` will stop the training and save
the results to `mnist.model` file, which then can be loaded by
`examples/mnist_sdl.ml` to visualize the model.

### Keybindings

You can use following keys to manipulate the demo:

| Key           | Action                                          |
|---------------|-------------------------------------------------|
| `Arrow Left`  | Select previous image from the verification set |
| `Arrow Right` | Select next     image from the verification set |
| `Arrow Up`    | Select random   image from the verification set |
| `Arrow Down`  | Clear image                                     |
| `Mouse Left`  | Draw on image                                   |
| `Mouse Right` | Erase from image                                |

## Algorithms

Currently the two supported methods of learning are:

 - using [finite differences](https://en.wikipedia.org/wiki/Finite_difference),
 - using [back propagation](https://en.wikipedia.org/wiki/Backpropagation).

## Activation functions

Implemented activation functions to choose from are:

 - [Sigmoid](https://en.wikipedia.org/wiki/Sigmoid_function),
 - [ReLU](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)).
