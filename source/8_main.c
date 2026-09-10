#include "../headers/8_main.h"

int main(void)
{
    srand(42);
    dataset_t test_data = malloc(sizeof(dataset));
    dataset_t train_data = malloc(sizeof(dataset));
    if (!test_data || !train_data) { // checking if the mallocs went well
        perror("malloc fot datasets didn't work in main");
        return 1;
    }
    
    int test_ret = load_dataset(test_data, "data/test_labels_idx1",
        "data/test_images_idx3");
    int train_ret = load_dataset(train_data, "data/train_labels_idx1",
        "data/train_images_idx3");

    if (test_ret || train_ret) {
        perror("error loading data");
        destroy_dataset(&train_data);
        destroy_dataset(&test_data);
    }

    for (int i = 0; i < 10; i++) {
        print_image(test_data, rand() % (test_data->nr_samples + 1));
        print_image(train_data, rand() % (train_data->nr_samples + 1));
    }

    destroy_dataset(&test_data);
    destroy_dataset(&train_data);
    return 0;
}
