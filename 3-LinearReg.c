#include <stdio.h>
#include "MyFile.h"

int main() {
    int n_samples = 4;
    int n_features = 2;

    // Datos de ejemplo para regresión lineal
    double X[4][2] = {
        {1.0, 2.0},
        {2.0, 3.0},
        {3.0, 4.0},
        {4.0, 5.0}
    };
    double y[4] = {3.0, 5.0, 7.0, 9.0};

    // Inicializar pesos para regresión lineal
    double weights[2] = {0.0, 0.0};

    // Entrenar el modelo de regresión lineal
    train_linear_regression(X, y, weights, n_samples, n_features);

    // Hacer una predicción con el modelo de regresión lineal
    double new_data[2] = {5.0, 6.0};
    double prediction = predict_linear_regression(new_data, weights, n_features);
    printf("Prediccion de Regresion Lineal: %f\n", prediction);

    return 0;
}
