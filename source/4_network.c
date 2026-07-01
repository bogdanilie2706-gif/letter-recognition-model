#include "../headers/4_network.h"

network_t create_network()
{
    network_t net = malloc(sizeof(network_size));
    if (!net) {
        perror("couldn't allocate net in create_network");
        return NULL;
    }
    net->head = net->tail = NULL;
    net->nr_layers = 0;
    return net
}

void free_network(network_t *net)
{

}

void network_add_layer(network_t net, int input_size, int output_size,
	float (*activation_func)(float), float (*activation_derivative)(float))
{

}

matrix_t network_forward(network_t net, matrix_t input)
{

}

void network_backward(network_t net, matrix_t grad)
{

}

void network_update(network_t net, float learning_rate)
{

}
