#include <stdio.h>
#include "MyFile.h"

int main() {
    int n_samples = 4;
    int n_features = 2;

    // Datos de ejemplo para SVM
    DataPoint data[4] = {
        {(double[]){1.0, 2.0}, 1},
        {(double[]){2.0, 3.0}, 1},
        {(double[]){3.0, 4.0}, -1},
        {(double[]){4.0, 5.0}, -1}
    };

    // Inicializar pesos para SVM
    double weights[2] = {0.0, 0.0};

    // Entrenar el modelo SVM con kernel lineal
    train_svm(data, weights, n_samples, n_features, LINEAR);

    // Imprimir los pesos después del entrenamiento
    printf("Pesos después del entrenamiento:\n");
    for (int i = 0; i < n_features; i++) {
        printf("Peso %d: %f\n", i, weights[i]);
    }

    // Hacer una predicción con el modelo SVM
    double new_data[2] = {3.0, 3.5};
    double prediction = predict_svm(new_data, weights, n_features, LINEAR);

    printf("Prediccion de SVM: %f\n", prediction);
    printf("Clasificacion: %s\n", prediction >= 0 ? "Clase 1" : "Clase -1");

    return 0;
}
