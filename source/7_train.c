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

void fill_batches(dataset_t df, batches_t batches, int *index)
{
	for (int i = 0; i < batches->nr_batches; i++) {
		
		fill_matrix(batches->target[i], 0.0f); // makes the whole target batch 0.0f
		for (int j = 0; j < batches->target[i]->rows; j++) {
			
			int pos = df->labels[index[i * batches->target[i]->rows + j]] - 1;

			batches->target[i]->data[j * batches->target[i]->cols + pos] = 1.0f; 
			// sets only the row with the target letter to 1.0f in that column
		}

		for (int j = 0; j < batches->input[i]->rows; j++) {
			for (int k = 0; k < batches->input[i]->cols; k++)
				batches->input[i]->data[j * batches->input[i]->cols + k] = 
					df->images[index[i * batches->input[i]->rows + j]][k] / 255.0f;
		}// transforms each pixel to a value between 0 and 1 and puts it in the batch as input
	}
}

batches_t create_batches_struct(int nr_batches, int input_cols, int target_cols, int batch_size)
{
	batches_t batches = malloc(sizeof(batches_struct));
	if (!batches) {
		perror("couldn't allocate batches struct in create_batches_struct");
		return NULL;
	}
	batches->input = malloc(sizeof(matrix_t) * nr_batches); // allocates the matrix_t
	batches->target = malloc(sizeof(matrix_t) * nr_batches); // vectors for input and target

	if (!batches->input || !batches->target) {
		perror("couldn't allocate batches input or target in create_batches_struct");
		destroy_batches_struct(&batches);
		return NULL;
	}

	for (int i = 0; i < nr_batches; i++) {
		batches->input[i] = create_matrix(batch_size, input_cols); // creates each matrix
		batches->target[i] = create_matrix(batch_size, target_cols); // in both vectors

		if (!batches->input[i] || !batches->target[i]) {
			perror("couldn't allocate batches input[i] or target[i] in create_batches_struct");
			destroy_batches_struct(&batches);
			return NULL;
		}
	}
	batches->nr_batches = nr_batches;

	return batches;
}

void destroy_batches_struct(batches_t *batches)
{
	if (!*batches)
		return;

	for (int i = 0; i < (*batches)->nr_batches; i++) {
		free_matrix(&(*batches)->input[i]);
		free_matrix(&(*batches)->target[i]); // frees each matrix
	}
	free((*batches)->input);
	free((*batches)->target); // frees the matrix vector itself
	free(*batches); // frees the structure
	*batches = NULL;
}

void shuffle_batches(dataset_t df, batches_t batches, int *index)
{
	shuffle_int_vector(index, df->nr_samples);
	fill_batches(df, batches, index);
}

static int good_guess_counter(matrix_t output, matrix_t target)
{
	int good_guesses = 0;
	for (int k = 0; k < output->rows; k++) {
		float max = output->data[k * output->cols]; // first ellem
		int pos = 0;

		for (int l = 1; l < output->cols; l++) {
			if (max < output->data[k * output->cols + l]) {
				max = output->data[k * output->cols + l]; // searches for each row
				pos = l;
			}
		}
		if (target->data[k * output->cols + pos] > 0.5f)
			good_guesses++; // used > 0.5f in case of any problem with comparasion between floats
	}
	return good_guesses;
}

static void print_epoch_info(FILE *file, int epoch_nr, float loss, int train_good_guesses, int test_good_guesses, int train_nr_images, int test_nr_images)
{
		fprintf(file, "\n---------- epoch number %d ----------\n", epoch_nr);
		fprintf(file, "Loss: %.6f\n", loss);
		
		fprintf(file, "Right guesses made in train: %d/%d\n", train_good_guesses, train_nr_images);
		fprintf(file, "Percentage of train right guesses: %.6f%%\n", (float)((float)train_good_guesses/(float)train_nr_images) * 100.0f);

		fprintf(file, "Right guesses made in test: %d/%d\n", test_good_guesses, test_nr_images);
		fprintf(file, "Percentage of test right guesses: %.6f%%\n", (float)((float)test_good_guesses/(float)test_nr_images) * 100.0f);
		fprintf(file, "---------------------------------------\n");
}

void train_letter_model(dataset_t train, dataset_t test, network_t net, int batch_size, float learning_rate, int nr_epochs)
{

	int train_nr_batches = train->nr_samples / batch_size;
	int test_nr_batches = test->nr_samples / batch_size;

	int *index = malloc(sizeof(int) * train_nr_batches * batch_size);
	if (!index) { // checking the allocation
		perror("couldn't allocate the index vector in train_letter_model");
		return;
	} // index vector for easier shuffling each epoch

	batches_t train_batches = create_batches_struct(train_nr_batches, 784, 26, batch_size);
	batches_t test_batches = create_batches_struct(test_nr_batches, 784, 26, batch_size);
	if (!train_batches || !test_batches) {
		perror("batches couldn't be allocated in train_letter_model");
		destroy_batches_struct(&train_batches);
		destroy_batches_struct(&test_batches);
		return;
	}

	for (int i = 0; i < train_nr_batches * batch_size; i++)
		index[i] = i;

	fill_batches(train, train_batches, index);
	fill_batches(test, test_batches, index); 
	// we can use the same index vector since its still not shuffled

	matrix_t output, grad;
	for (int j = 0; j < nr_epochs; j++) {

		printf("Starting epoch nr %d \n", j + 1);

		// train section
		int train_good_guesses = 0;
		for (int i = 0; i < train_nr_batches; i++) {
			// printf("starting batch nr %d out of %d\n", i + 1, train_nr_batches);

			output = network_forward(net, train_batches->input[i]); // output free is handled by layer_forward func
			grad = cross_entropy_gradient(output, train_batches->target[i]);
			train_good_guesses += good_guess_counter(output, train_batches->target[i]);
			network_backward(net, &grad); // grad is freed inside the network_backward func
			network_update(net, learning_rate);

			// printf("ended the batch with succes\n\n");
		}

		printf("Training finished, starting testing\n");

		// test section
		int test_good_guesses = 0;
		float loss = 0.0f;
		for (int i = 0; i < test_nr_batches; i++) {
			// printf("starting batch nr %d out of %d\n", i + 1, test_nr_batches);

			output = network_forward(net, test_batches->input[i]);
			test_good_guesses += good_guess_counter(output, test_batches->target[i]);
			loss += cross_entropy_loss(output, test_batches->target[i]);

			// printf("ended the batch with succes\n\n");
		}
		loss /= test_nr_batches;

		printf("Testing finished\n");

		FILE *file = fopen("model_epoch_output.txt", "a");
		print_epoch_info(file, j + 1, loss, train_good_guesses, test_good_guesses, train_nr_batches * batch_size, test_nr_batches * batch_size);
		fclose(file);

		if (j < nr_epochs - 1)
			shuffle_batches(train, train_batches, index);
	}
	printf("training has ended\n");

	free(index);
	destroy_batches_struct(&test_batches);
	destroy_batches_struct(&train_batches);
}
