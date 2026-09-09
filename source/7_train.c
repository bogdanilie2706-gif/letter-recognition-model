#include "../headers/7_train.h"

static void shuffle_int_vector(int *vector, int nr_values)
{
	for (int i = nr_values - 1; i >= 0; i--) {
		int value = rand() % (i + 1);
		int aux = vector[i];
		vector[i] = vector[value];
		vector[value] = aux;
	}
	// the shuffle that will be done at the start of every epoch
}

static void images_to_batches(dataset_t df, matrix_t *batches, int *index, int nr_batches)
{
	for (int i = 0; i < nr_batches; i++) {
		for (int j = 0; j < batches[i]->rows; j++) {
			for (int k = 0; k < batches[i]->cols; k++)
				batches[i]->data[j * batches[i]->cols + k] = df->images[index[i * batches[i]->rows + j]][k] / 255.0f;
		}
	}
}

batches_t create_batches_struct(dataset_t dataset, int nr_batches, int input_cols, int target_cols)
{
	batches_t batches = malloc(sizeof(batches_struct));
	if (!batches) {
		perror("couldn't allocate batches struct in create_batches_struct");
		return NULL;
	}
	batches->input = malloc(sizeof(matrix_t) * nr_batches);
	batches->target = malloc(sizeof(matrix_t) * nr_batches);

	if (!batches->input || !batches->target) {
		perror("couldn't allocate batches input or target in create_batches_struct");
		destroy_batches_struct(&batches);
		return NULL;
	}

	for (int i = 0; i < nr_batches; i++) {
		batches->input[i] = create_matrix(nr_batches, input_cols);
		batches->target[i] = create_matrix(nr_batches, target_cols);

		if (!batches->input[i] || !batches->target[i]) {
			perror("couldn't allocate batches input[i] or target[i] in create_batches_struct");
			destroy_batches_struct(&batches);
			return NULL;
		}
	}

	return batches;
}

void destroy_batches_struct(batches_t *batches)
{
	if (!*batches)
		return;

	for (int i = 0; i < (*batches)->nr_batches; i++) {
		free_matrix((*batches)->input);
		free_matrix((*batches)->target);
	}
	free((*batches)->input);
	free((*batches)->target);
	free(*batches);
	*batches = NULL;
}

void shuffle_batches(dataset_t df, batches_t batches, int *index)
{

}

void train_letter_model(dataset_t train, dataset_t test, network_t net, float learning_rate)
{
	int *index = malloc(sizeof(int) * train->nr_samples);
	if (!index) { // checking the allocation
		perror("couldn't allocate the index vector in train_letter_model");
		return;
	}
	// index vector for easier shuffling each epoch
	for (int i = 0; i < train->nr_samples; i++)
		index[i] = i;




	free(index);
}
