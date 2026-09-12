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
    
    int test_ret = load_dataset(test_data, "data/test_labels_idx1","data/test_images_idx3");
    int train_ret = load_dataset(train_data, "data/train_labels_idx1","data/train_images_idx3");
    if (test_ret || train_ret) {
        perror("error loading data in main");
        destroy_dataset(&train_data);
        destroy_dataset(&test_data);
        return 1;
    }

    network_t net = create_network();
    network_add_layer(net, 28 * 28, 128, ACTIVATION_RELU);
    network_add_layer(net, 128, 64, ACTIVATION_RELU);
    network_add_layer(net, 64, 26, ACTIVATION_SOFTMAX);
    network_init_weights(net);

    train_letter_model(train_data, test_data, net, 32, 0.004, 1);

    save_model(net, "le_bot_robot");
    free_network(&net);
    net = load_model("le_bot_robot");

    train_letter_model(train_data, test_data, net, 32, 0.004, 1);

    save_model(net, "ze_bot");

    free_network(&net);
    destroy_dataset(&test_data);
    destroy_dataset(&train_data);
    return 0;
}
