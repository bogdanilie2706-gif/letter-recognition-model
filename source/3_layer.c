#include "../headers/3_layer.h"

layer_t create_layer(int input_size, int output_size,
					  float (*activation_func)(float),
					  float (*activation_derivative)(float))
{
	layer_t layer = malloc(sizeof(layer_size));
	if (!layer) { // checking allocation for layer
		perror("couldn't allocate layer_t structure in layer_create");
		return NULL;
	}

	// they will be allocated during the training
	layer->z = layer->activation = layer->weights_grad = layer->bias_grad = NULL;
	layer->next = layer->prev = NULL;

	// creates the matrices that will be used 
	layer->weights = create_matrix(input_size, output_size);
	layer->bias = create_matrix(1, output_size); 
	if (!layer->weights || !layer->bias) {
		perror("couldn't create the matrices for weights or biases in layer_create");
		free_layer(&layer); // only frees allocated blocks
		return NULL;
	}

	// functions that will be used for this layer in training
	layer->activation_func = activation_func;
	layer->activation_derivative = activation_derivative;
	
	layer->input_size = input_size;
	layer->output_size = output_size;

	return layer;
}

void free_layer(layer_t *layer)
{	
	if (!*layer) return;

	layer_t aux = *layer;
	free_matrix(&aux->weights);
	free_matrix(&aux->bias);
	free_matrix(&aux->z);
	free_matrix(&aux->activation);
	free_matrix(&aux->weights_grad);
	free_matrix(&aux->bias_grad);

	free(*layer);
	*layer = NULL;
}

void layer_init_weights(layer_t layer, float min, float max)
{
	matrix_random_init(layer->weights, min, max);
}

matrix_t layer_forward(layer_t layer, matrix_t input)
{
	if (layer->z || layer->activation) {
		free_matrix(&layer->z);
		free_matrix(&layer->activation);
	}
	matrix_t temp = multiply_matrix(input, layer->weights);
	layer->z = matrix_add_bias(temp, layer->bias);
	free_matrix(&temp);
	layer->activation = apply_matrix(layer->z, layer->activation_func);

	return layer->activation;
}

matrix_t layer_backward(layer_t layer, matrix_t input, matrix_t grad_output)
{


}

void layer_update(layer_t layer, float learning_rate)
{

}
