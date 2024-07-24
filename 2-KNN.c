#include <stdio.h>
#include "MyFile.h"

int main() {
    int n_samples = 4;
    int n_features = 2;
    int k = 3;

    // Datos de ejemplo para KNN
    DataPoint data[4] = {
        {(double[]){0.5, 1.2}, 0},
        {(double[]){1.3, 3.1}, 1},
        {(double[]){2.1, 2.9}, 1},
        {(double[]){0.7, 0.8}, 0}
    };

    // Nueva muestra para clasificar con KNN
    double new_data_knn[2] = {0.4, 1.0};

    // Mostrar datos a los que se parece y el número de vecinos k
    printf("Nueva muestra para clasificar: (%.2f, %.2f)\n", new_data_knn[0], new_data_knn[1]);
    printf("Numero de vecinos (k): %d\n", k);

    // Array para almacenar las etiquetas de los k vecinos más cercanos
    int neighbors[k];

    // Encontrar los k vecinos más cercanos e imprimir información
    find_k_nearest_neighbors(data, new_data_knn, n_samples, n_features, k, neighbors);

    // Clasificación con KNN
    int label_knn = classify_knn(data, new_data_knn, n_samples, n_features, k);
    printf("Prediccion de KNN: %d\n", label_knn);

    return 0;
}
