#include <stdio.h>
#include <stdlib.h>
#include "MyANN.h"

int main() {
    // Datos de entrenamiento: Celsius a Fahrenheit
    float celsius[] = {-10.0, -9.0, -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 10.0, 20.0, 24.0};
    float fahrenheit[] = {14.0, 15.8, 17.6, 19.4, 21.2, 23.0, 24.8, 26.6, 28.4, 30.2, 32.0, 50.0, 68.0, 75.2};

    // Crear la arquitectura de la red neuronal
    int neurons_per_layer[] = {1, 10, 1};  // 1 neurona en la capa de entrada, 10 en la oculta, 1 en la de salida
    neural_network nn = create_architecture(3, neurons_per_layer);//HASTA AQUI CREA Y FUNCIONA TODO BIEN

    // Entrenar la red neuronal
    train(&nn, celsius, fahrenheit, 14, 0.01, 10);

    // Hacer predicciones
    float test_input_0 = 0.0;  // Predicción para 0 grados Celsius
    float test_input_24 = 24.0;  // Predicción para 24 grados Celsius

    printf("Predicción para 0 grados Celsius:\n");
    predict(&nn, &test_input_0);

    printf("Predicción para 24 grados Celsius:\n");
    predict(&nn, &test_input_24);

    return 0;
}
