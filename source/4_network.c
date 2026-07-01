#include "../headers/4_network.h"

network_t create_network()
{
    network_t net = malloc(sizeof(network_size));
    if (!net) { // checking allocation
        perror("couldn't allocate net in create_network");
        return NULL;
    }
    net->head = net->tail = NULL;
    net->nr_layers = 0;
    return net;
}

void free_network(network_t *net)
{
    layer_t crt = (*net)->head;
    while (crt) { // freeing every layer in the list
        layer_t aux = crt->next;
        free_layer(crt);
        crt = aux;
    }
    free(*net);
    *net = NULL;
}

void network_add_layer(network_t net, int input_size, int output_size,
	float (*activation_func)(float), float (*activation_derivative)(float))
{
    if (!net) { // checking if net is initialized
        perror("couldn't add layer in network_add_layer, net was NULL");
        return;
    }

    layer_t layer = create_layer(input_size, output_size,
                    activation_func, activation_derivative);
    if (!layer) { // checking allocation
        perror("layer couldn't be created in network_add_layer");
        return;
    }

    if (!net->head) { // if the list is empty
        net->head = layer;
        net->tail = layer;
    } else { // if there is one or more layers in the list
        layer->prev = net->tail;
        net->tail->next = layer;
        net->tail = layer;
    }
    net->nr_layers++;
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
