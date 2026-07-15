#include "../headers/7_train.h"

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
	
	for (int i = train->nr_samples - 1; i >= 0; i--) {
		int value = rand() % (i + 1);
		int aux = index[i];
		index[i] = index[value];
		index[value] = aux;
	} // the shuffle that will be done at the start of every epoch

	free(index);
}
