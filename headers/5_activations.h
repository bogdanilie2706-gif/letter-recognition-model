#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H



#include "../headers/2_matrix.h"

// used for hidden layers in layer_forward
matrix_t relu(matrix_t z);

// used for hidden layers in layer_backward
matrix_t relu_derivative(matrix_t z);

matrix_t leaky_relu(matrix_t z);

matrix_t leaky_relu_derivative(matrix_t z);

matrix_t tanh_activation(matrix_t z);

matrix_t tanh_derivative(matrix_t z);

matrix_t sigmoid(matrix_t z);

matrix_t sigmoid_derivative(matrix_t z);

// used for the 'output' layer (last one) in layer_forward
matrix_t softmax(matrix_t z);

// used for the 'output' layer (last one) in layer_backward
// placeholder for sofmax derivative
matrix_t identity(matrix_t z);

#endif