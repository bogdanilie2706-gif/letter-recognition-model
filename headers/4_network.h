#ifndef NETWORK_H
#define NETWORK_H

#include "../headers/3_layer.h"

typedef struct network_struct {
    layer_t head; // the first layer from the input
    layer_t tail; // the last layer from the output
    int nr_layers;
} network_struct, *network_t; 

// initializez the network, makes the head and tail NULL 
// and the nr_layer 0
network_t create_network();

// frees the entire network along with the structure itself
// and sets the value to NULL
void free_network(network_t *net);

// appends a new layer to the tail of the network
void network_add_layer(network_t net, int input_size, int output_size,
	float (*activation_func)(float), float (*activation_derivative)(float));

// takes the input and returns the output made by the network
matrix_t network_forward(network_t net, matrix_t input);

// creates the weights and biases gradients trough 
// backprop based on the grad given by prediction - target
// also frees the grad matrix inside so the caller doesn't have to
void network_backward(network_t net, matrix_t *grad);

// updates the weights and biases with the 
// gradients calculated in each layer
void network_update(network_t net, float learning_rate);


#endif
