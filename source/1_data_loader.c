#include "../headers/1_data_loader.h"

int load_dataset(dataset_t data, char *idx1_filename, char *idx3_filename)
{
	FILE *idx1 = fopen(idx1_filename, "rb");
	FILE *idx3 = fopen(idx3_filename, "rb");
	if (!idx1 || !idx3) {
		perror("couldn't open files in load_dataset");
		return 1;
	}
	int idx1_ret = load_idx1_file(idx1, data);
	int idx3_ret = load_idx3_file(idx3, data);

	if(idx1_ret || idx3_ret) {
		perror("error loading the data in the files in load_dataset");
		fclose(idx1);
		fclose(idx3);
		return 1;
	}

	fclose(idx1);
	fclose(idx3);
	return 0;
}


int load_idx3_file(FILE *idx, dataset_t data)
{
	unsigned char buffer[16]; // for reading the header
	int x = fread(buffer, 4, 4, idx);
	if (x != 4) {
		perror("fread at buffer in load_idx3_file stopped early");
		return 1;
	}

	int n = ((unsigned int)buffer[4] << 24) | 
			((unsigned int)buffer[5] << 16) |
			((unsigned int)buffer[6] << 8) |
			((unsigned int)buffer[7]); // conversion from big to little endian

	printf("nr images = %d\n", n);

	data->nr_samples = n;
	data->images = malloc(sizeof(unsigned char *) * n);
	if (!data->images) {
		perror("data->images couldn't be allocated in load_idx3_file");
		return 1;
	}

	int ok = 1, i;
	char aux[784]; // for reading the transposed image
	for (i = 0; i < n; i++) {
		
		data->images[i] = malloc(sizeof(unsigned char) * 784);
		if (!data->images[i]) {
			ok = 0;
			perror("data->images[i] for writing coldn't be allocated in load_idx3_file");
			break;
		}

		x = fread(aux, 1, 784, idx);
		if (x != 784) {
			ok = 0;
			perror("fread at data->images[i] in load_idx3_file stopped early");
			break;
		}

		for (int j = 0; j < 28; j++) {
			for (int k = 0; k < 28; k++)
				data->images[i][k * 28 + j] = aux[j * 28 + k];
				// transposing the image
		}
	}

	if(!ok) { // checking if there was a problem in the for loop
		for (int j = 0; j < i; j++)
			free(data->images[j]);
		free(data->images);
		printf("%d\n", i);
		return 1;
	}

	return 0;
}

int load_idx1_file(FILE *idx, dataset_t data)
{
	unsigned char buffer[8]; // for reading the header
	int x = fread(buffer, 4, 2, idx);
	if (x != 2) {
		perror("fread at buffer in load_idx1_file stopped early");
		return 1;
	}

	int n = ((unsigned int)buffer[4] << 24) | 
			((unsigned int)buffer[5] << 16) |
			((unsigned int)buffer[6] << 8) |
			((unsigned int)buffer[7]); // conversion from big to little endian
	data->nr_samples = n;

	printf("nr_labels = %d\n", n);

	data->labels = malloc(sizeof(char) * n);
	if (!data->labels) { // checking the malloc
		perror("data->labels in load_idx1_file couldn't be allocated");
		return 1;
	}

	x = fread(data->labels, 1, n, idx);
	if (x != n) { // checking if fread stopped
		free(data->labels);
		perror("fread at labels in load_idx1_file stopped early");
		return 1;
	}

	return 0;
}

void print_image(dataset_t data, int image_index)
{
	printf("the letter %c is in the image\n", 'a' + 1 - data->labels[image_index]);
	for (int i = 0; i < 28; i++) {
		for (int j = 0; j < 28; j++)
		 printf("%3hhu ", data->images[image_index][i * 28 + j]);
	printf("\n");
	}
	printf("\n");
}

void destroy_dataset(dataset_t *data)
{
	dataset_t aux = *data;
	if (aux->labels)
		free(aux->labels);
		
	if (aux->images) {
		for (int i = 0; i < aux->nr_samples; i++)
			free(aux->images[i]);
		free(aux->images);
	}
	free(*data);
}


