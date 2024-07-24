#include <stdio.h>
#include "MyFile.h"

int main() {
    int n_samples = 4;
    int n_features = 2;

    // Datos de ejemplo para regresi�n log�stica
    double X[4][2] = {
        {0.5, 1.2},
        {1.3, 3.1},
        {2.1, 2.9},
        {0.7, 0.8}
    };
    double y[4] = {0, 1, 1, 0};

    // Inicializar pesos para regresi�n log�stica
    double weights[2] = {0, 0};

    // Entrenar el modelo de regresi�n log�stica
    train_logistic_regression(X, y, weights, n_samples, n_features);

    // Hacer una predicci�n con el modelo de regresi�n log�stica
    double new_data_logistic[2] = {1.0, 2.0};
    double prediction_logistic = predict_logistic_regression(new_data_logistic, weights, n_features);
    printf("Predicci�n de Regresi�n Log�stica: %f\n", prediction_logistic);

    // Datos de ejemplo para KNN
    DataPoint data[4] = {
        {(double[]){0.5, 1.2}, 0},
        {(double[]){1.3, 3.1}, 1},
        {(double[]){2.1, 2.9}, 1},
        {(double[]){0.7, 0.8}, 0}
    };

    // Nueva muestra para clasificar con KNN
    double new_data_knn[2] = {1.0, 2.0};
    int k = 3;

    // Clasificaci�n con KNN
    int label_knn = classify_knn(data, new_data_knn, n_samples, n_features, k);
    printf("Predicci�n de KNN: %d\n", label_knn);

    return 0;
}
