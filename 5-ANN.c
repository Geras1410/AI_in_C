#include <stdio.h>
#include "MyFile.h"

#define N_SAMPLES 5

int main() {
    int n_layers = 3;
    int neurons_per_layer[] = {1, 5, 1};
    NeuralNetwork nn = initialize_nn(n_layers, neurons_per_layer);

    // Datos de entrenamiento: Celsius y sus respectivos valores en Fahrenheit
    double X[N_SAMPLES] = {0.0, 10.0, 20.0, 30.0, 40.0};
    double y[N_SAMPLES] = {32.0, 50.0, 68.0, 86.0, 104.0};

    // Entrenamiento
    double learning_rate = 0.01;
    int epochs = 10000;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        for (int i = 0; i < N_SAMPLES; ++i) {
            double input[] = {X[i]};
            double output[] = {y[i]};
            back_propagation(&nn, input, output, learning_rate);
        }
    }

    // Predicción
    double test_input[] = {25.0};
    forward_propagation(&nn, test_input);
    double prediction = nn.layers[n_layers - 1].a[0];
    printf("Predicción para 25 grados Celsius: %f Fahrenheit\n", prediction);

    // Mostrar los pesos finales
    print_weights(&nn);

    // Liberar memoria
    for (int i = 0; i < n_layers; ++i) {
        free(nn.layers[i].z);
        free(nn.layers[i].a);
        free(nn.layers[i].b);
        if (i > 0) {
            for (int j = 0; j < nn.layers[i].n_neurons; ++j) {
                free(nn.layers[i].w[j]);
            }
            free(nn.layers[i].w);
        }
    }
    free(nn.layers);

    return 0;
}
