#ifndef LAYER_H
#define LAYER_H

#include "headers/2_matrix.h"

typedef struct layer_struct {
	int input_size;
	int output_size;

	matrix_t weights;       // (input_size x output_size)
	matrix_t bias;          // (1 x output_size), one bias per output neuron

	matrix_t z;             // raw output before activation (batch_size x output_size), needed for backprop
	matrix_t activation;    // output after activation (batch_size x output_size), fed to next layer

	matrix_t weights_grad;  // gradient of loss w.r.t. weights, computed during backward pass
	matrix_t bias_grad;     // gradient of loss w.r.t. bias, computed during backward pass

	float (*activation_func)(float);       // activation function applied elementwise to z
	float (*activation_derivative)(float); // derivative of activation_func, needed for backprop

	struct layer *next;
	struct layer *prev;
} layer_size, *layer_t;

// allocates a layer with given input/output sizes, sets up weights and bias matrices
// does NOT randomly initialize weights, call layer_init_weights function for that
layer_t layer_create(int input_size, int output_size,
					  float (*activation_func)(float),
					  float (*activation_derivative)(float));

// frees the whole layer_t structure along with the matrices
// does NOT touch next/prev layers, caller is responsible for unlinking first
void layer_free(layer_t *l);

// randomly initializes weights and bias for this layer
// caller decides the range
void layer_init_weights(layer_t l, float min, float max);

// computes z = input * weights + bias, then activation = activation_func(z), elementwise
// input is a (batch_size x input_size) matrix
// stores both z and activation inside the layer (needed later for backward pass)
// returns the activation matrix (also accessible via l->activation)
matrix_t layer_forward(layer_t l, matrix_t input);

/* --- backward pass (for later, once train.c is being designed) --- */

// computes weights_grad, bias_grad, and returns the gradient to propagate
// to the previous layer (gradient w.r.t. this layer's input)
// grad_output is the gradient of the loss w.r.t. this layer's activation
// input is the original input this layer received during forward pass
matrix_t layer_backward(layer_t l, matrix_t input, matrix_t grad_output);

/* --- Weight update --- */

// updates weights and bias using their stored gradients and a learning rate
// weights -= learning_rate * weights_grad, same for bias
void layer_update(layer_t l, float learning_rate);

#endif /* LAYER_H */