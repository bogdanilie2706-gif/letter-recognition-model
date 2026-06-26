#include "../headers/1_data_loader.h"

idx3_t load_idx3_file(FILE *idx)
{
	char buffer[16]; // for reading the header
	int x = fread(buffer, 4, 4, idx);
	if (x != 4) {
		perror("fread at buffer in load_idx3_file stopped early");
		return NULL;
	}

	idx3_t data = malloc(sizeof(idx3));
	if (!data) {
		perror("data couldn't be allocated in load_idx3_file");
		return NULL;
	}

	int n = ((int)buffer[4] << 24) | 
			((int)buffer[5] << 16) |
			((int)buffer[6] << 8) |
			((int)buffer[7]); // conversion from big to little endian

	data->nr_images = n;
	// printf()
	data->images = malloc(sizeof(unsigned char *) * n);
	if (!data->images) {
		free(data);
		perror("data->images couldn't be allocated in load_idx3_file");
		return NULL;
	}

	int ok = 1, i;
	for (i = 0; i < n; i++) {
		
		data->images[i] = malloc(sizeof(unsigned char) * 784);
		if (!data->images[i]) {
			ok = 0;
			perror("data->images[i] for writing coldn't be allocated in load_idx3_file");
			break;
		}

		x = fread(data->images[i], 1, 784, idx);
		if (x != 784) {
			ok = 0;
			perror("fread at data->images[i] in load_idx3_file stopped early");
			break;
		}
	}

	if(!ok) {
		for (int j = 0; j < i; j++)
			free(data->images[j]);
		free(data->images);
		free(data);
		printf("%d\n", i);
		return NULL;
	}

	return data;
}
