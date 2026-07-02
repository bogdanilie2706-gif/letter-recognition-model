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
	if (layer->z) 			free_matrix(&layer->z);
	if (layer->activation) 	free_matrix(&layer->activation);
	if (layer->input)      	free_matrix(&layer->input);
	// frees the matrices if they have been used before

	layer->input = copy_matrix(input); // will be used in backprop
	matrix_t temp = multiply_matrix(input, layer->weights);
	if (!layer->input || !temp) {
		perror("temp and input couldn't be allocated by copy and mul in layer_forward");
		return NULL;
	}
	layer->z = matrix_add_bias(temp, layer->bias);
	free_matrix(&temp);
	layer->activation = apply_matrix(layer->z, layer->activation_func);
	if (!layer->z || !layer->activation) {
		perror("z or activation couldn't be allocated by add bias or apply in layer_forward");
		return NULL;
	}

	return layer->activation;
}

matrix_t layer_backward(layer_t layer, matrix_t grad_output)
{
	// the matrices are freed if they have been already used
	if (layer->weights_grad) free_matrix(&layer->weights_grad);
	if (layer->bias_grad)    free_matrix(&layer->bias_grad);

	// chain rule: gradient w.r.t. z (pre-activation output)
	matrix_t deriv = apply_matrix(layer->z, layer->activation_derivative);
	matrix_t grad_z = matrix_elementwise_multiply(grad_output, deriv);
	if (!grad_z || !deriv) {
		perror("grad_z or deriv couldn't be allocated by apply and elem_mul in layer_backward");
		return NULL;
	}
	free_matrix(&deriv);

	// building the weights gradient
	matrix_t input_tran = transpose_matrix(layer->input);
	layer->weights_grad = multiply_matrix(input_tran, grad_z);
	if (!input_tran || !layer->weights_grad) {
		perror("input_tran or weights_grad couldn't be allocated by transpose and mul in layer_backward");
		return NULL;
	}
	free_matrix(&input_tran);

	// building the bias gradient
	layer->bias_grad = matrix_sum_rows(grad_z);
	if (!layer->bias_grad) {
		perror("bias_grad couldn't be allocated by sum_rows in layer_backward");
		return NULL;
	}

	// building the gradient for the next layer
	matrix_t weights_tran = transpose_matrix(layer->weights);
	matrix_t back_grad = multiply_matrix(grad_z, weights_tran); 
	// grad_z: (batch x output_size) * weights_T: (output_size x input_size)
	// result: (batch x input_size)
	if (!weights_tran || !back_grad) {
		perror("weights_tran or back_grad couldn't be allocated by tran and mul in layer_backward");
		return NULL;
	}

	free_matrix(&weights_tran);
	free_matrix(&grad_z);

	return back_grad;
}

void layer_update(layer_t layer, float learning_rate)
{
	matrix_t temp = matrix_scalar_multiply(layer->weights_grad, learning_rate);
	if (!temp) {
		perror("temp couldn't be allocated by scalar_mul for weights_grad in layer_update");
		return;
	}
	matrix_t aux = layer->weights;
	layer->weights = temp;
	free(aux);

	temp = matrix_scalar_multiply(layer->bias_grad, learning_rate);
	if (!temp) {
		perror("temp couldn't be allocated by scalar_mul for bias_grad in layer_update");
		return;
	}
	aux = layer->bias;
	layer->bias = temp;
	free(aux);
}
