#include "../headers/4_network.h"

network_t create_network()
{
    network_t net = malloc(sizeof(network_struct));
    if (!net) { // checking allocation
        perror("couldn't allocate net in create_network");
        return NULL;
    }
    net->head = net->tail = NULL; // empty list
    net->nr_layers = 0;
    return net;
}

void free_network(network_t *net)
{
    layer_t crt = (*net)->head;
    while (crt) { // freeing every layer in the list
        layer_t aux = crt->next;
        free_layer(&crt);
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
    if (!net)
        return NULL; // edge case

    layer_t crt_layer = net->head; // starting from the 'input' layer
    matrix_t crt_input = input;
    while (crt_layer) {
        crt_input = layer_forward(crt_layer, crt_input);
        if(!crt_input) { // checking for null return
            perror ("crt_input is null by layer_forward in network_forward");
            return NULL;
        }
        crt_layer = crt_layer->next;
    }
    return crt_input; // returning the output
}

void network_backward(network_t net, matrix_t *grad)
{
    if (!net)
        return; // edge case

    layer_t crt_layer = net->tail; // starting from the 'output' layer
    matrix_t crt_grad = *grad; // will free the matrix from the caller
    while (crt_layer) {
        matrix_t new_grad = layer_backward(crt_layer, crt_grad);
        if (crt_grad == *grad)
            free_matrix(grad);
        else
            free_matrix(&crt_grad);
        crt_grad = new_grad;
        if (!crt_grad) { // checking for null return
            perror ("crt_grad is null by layer_backward in network_backward");
            return;
        }
        crt_layer = crt_layer->prev;
    }
    free_matrix(&crt_grad);
}

void network_update(network_t net, float learning_rate)
{
    if (!net)
        return; // edge case
        
    layer_t crt_layer = net->head;
    while (crt_layer) { // learning rate to be decided by caller
        layer_update(crt_layer, learning_rate);
        crt_layer = crt_layer->next;
    }
}
