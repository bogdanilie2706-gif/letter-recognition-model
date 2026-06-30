#include "headers/3_layer.h"

layer_t layer_create(int input_size, int output_size,
					  float (*activation_func)(float),
					  float (*activation_derivative)(float))
{
	layer_t layer = malloc(sizeof(layer_size));

}

void layer_free(layer_t *l)
{

}

void layer_init_weights(layer_t l, float min, float max)
{

}

matrix_t layer_forward(layer_t l, matrix_t input)
{

}

matrix_t layer_backward(layer_t l, matrix_t input, matrix_t grad_output)
{

}

void layer_update(layer_t l, float learning_rate)
{

}
