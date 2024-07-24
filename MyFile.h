#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define LEARNING_RATE 0.01
#define EPOCHS 1000
#define C 1.0  // Parámetro de regularización

typedef struct {
    double *features;
    int label;
} DataPoint;

typedef enum {
    LINEAR,
    POLYNOMIAL
} KernelType;

// Definición de las estructuras de datos
typedef struct {
    int n_neurons;
    double *z;
    double *a;
    double *b;
    double **w;
} Layer;

typedef struct {
    int n_layers;
    Layer *layers;
} NeuralNetwork;

// Funciones para Regresión Logística
double sigmoid(double z);
double sigmoid_prime(double z);
double relu(double z);
double relu_prime(double z);
double cross_entropy_loss(double y_true, double y_pred);
void train_logistic_regression(double X[][2], double y[], double weights[], int n_samples, int n_features);
double predict_logistic_regression(double X[], double weights[], int n_features);

// Funciones para K-Nearest Neighbors (KNN)
double euclidean_distance(double *a, double *b, int n_features);
void find_k_nearest_neighbors(DataPoint *data, double *input, int n_samples, int n_features, int k, int *neighbors);
int classify_knn(DataPoint *data, double *input, int n_samples, int n_features, int k);

// Funciones para Regresión Lineal
void train_linear_regression(double X[][2], double y[], double weights[], int n_samples, int n_features);
double predict_linear_regression(double X[], double weights[], int n_features);

//Funciones para SVM
void train_svm(DataPoint data[], double weights[], int n_samples, int n_features, KernelType kernel_type);
double predict_svm(double features[], double weights[], int n_features, KernelType kernel_type);

// Inicialización de la red neuronal
NeuralNetwork initialize_nn(int n_layers, int *neurons_per_layer);

// Propagación hacia adelante
void forward_propagation(NeuralNetwork *nn, double *input);

// Retropropagación y actualización de pesos
void back_propagation(NeuralNetwork *nn, double *input, double *output, double learning_rate);

// Función sigmoide
double sigmoid(double z) {
    return 1.0 / (1.0 + exp(-z));
}

double sigmoid_prime(double z) {
    double sig = sigmoid(z);
    return sig * (1.0 - sig);
}

double relu(double z) {
    return z > 0 ? z : 0;
}

double relu_prime(double z) {
    return z > 0 ? 1 : 0;
}

// Función de pérdida
double cross_entropy_loss(double y_true, double y_pred) {
    return - (y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred));
}

// Entrenamiento de la regresión logística
void train_logistic_regression(double X[][2], double y[], double weights[], int n_samples, int n_features) {
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        for (int i = 0; i < n_samples; i++) {
            double linear_model = 0.0;
            for (int j = 0; j < n_features; j++) {
                linear_model += X[i][j] * weights[j];
            }
            double y_pred = sigmoid(linear_model);
            double error = y_pred - y[i];

            for (int j = 0; j < n_features; j++) {
                weights[j] -= LEARNING_RATE * error * X[i][j];
            }
        }
    }
}

// Predicción
double predict_logistic_regression(double X[], double weights[], int n_features) {
    double linear_model = 0.0;
    for (int j = 0; j < n_features; j++) {
        linear_model += X[j] * weights[j];
    }
    return sigmoid(linear_model);
}

// Implementación de las funciones de K-Nearest Neighbors (KNN)

// Función para calcular la distancia euclidiana
double euclidean_distance(double *a, double *b, int n_features) {
    double distance = 0.0;
    for (int i = 0; i < n_features; i++) {
        distance += pow(a[i] - b[i], 2);
    }
    return sqrt(distance);
}

// Función para encontrar los k vecinos más cercanos
void find_k_nearest_neighbors(DataPoint *data, double *input, int n_samples, int n_features, int k, int *neighbors) {
    double *distances = (double *)malloc(n_samples * sizeof(double));
    int *indices = (int *)malloc(n_samples * sizeof(int));

    for (int i = 0; i < n_samples; i++) {
        distances[i] = euclidean_distance(data[i].features, input, n_features);
        indices[i] = i;
    }

    // Ordenar distancias y almacenar índices de los vecinos más cercanos
    for (int i = 0; i < n_samples - 1; i++) {
        for (int j = i + 1; j < n_samples; j++) {
            if (distances[i] > distances[j]) {
                double temp_dist = distances[i];
                distances[i] = distances[j];
                distances[j] = temp_dist;

                int temp_idx = indices[i];
                indices[i] = indices[j];
                indices[j] = temp_idx;
            }
        }
    }

    printf("Distancias y etiquetas de los %d vecinos mas cercanos:\n", k);
    for (int i = 0; i < k; i++) {
        int idx = indices[i];
        printf("Punto: (%.2f, %.2f), Distancia: %.3f, Etiqueta: %d\n",
               data[idx].features[0], data[idx].features[1], distances[i], data[idx].label);
        neighbors[i] = data[idx].label;
    }

    free(distances);
    free(indices);
}

// Función para clasificar basado en los vecinos más cercanos
int classify_knn(DataPoint *data, double *input, int n_samples, int n_features, int k) {
    int *neighbors = (int *)malloc(k * sizeof(int));
    find_k_nearest_neighbors(data, input, n_samples, n_features, k, neighbors);

    // Votación mayoritaria
    int count0 = 0, count1 = 0;
    for (int i = 0; i < k; i++) {
        if (neighbors[i] == 0) {
            count0++;
        } else {
            count1++;
        }
    }

    free(neighbors);
    return count1 > count0 ? 1 : 0;
}

// Implementación de la función de entrenamiento para la regresión lineal
void train_linear_regression(double X[][2], double y[], double weights[], int n_samples, int n_features) {
    double *gradients = (double *)malloc(n_features * sizeof(double));

    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        // Inicializar los gradientes
        for (int j = 0; j < n_features; j++) {
            gradients[j] = 0.0;
        }

        // Calcular los gradientes
        for (int i = 0; i < n_samples; i++) {
            double prediction = 0.0;
            for (int j = 0; j < n_features; j++) {
                prediction += X[i][j] * weights[j];
            }

            double error = prediction - y[i];

            for (int j = 0; j < n_features; j++) {
                gradients[j] += (2.0 / n_samples) * error * X[i][j];
            }
        }

        // Actualizar los pesos
        for (int j = 0; j < n_features; j++) {
            weights[j] -= LEARNING_RATE * gradients[j];
        }
    }

    free(gradients);
}

// Implementación de la función de predicción para la regresión lineal
double predict_linear_regression(double X[], double weights[], int n_features) {
    double prediction = 0.0;
    for (int j = 0; j < n_features; j++) {
        prediction += X[j] * weights[j];
    }
    return prediction;
}

// Función para calcular el producto punto
double dot_product(double *a, double *b, int n_features) {
    double result = 0.0;
    for (int i = 0; i < n_features; i++) {
        result += a[i] * b[i];
    }
    return result;
}

// Función para el kernel lineal
double linear_kernel(double *a, double *b, int n_features) {
    return dot_product(a, b, n_features);
}

// Función de pérdida para SVM con margen máximo
double hinge_loss(double prediction, int label) {
    return fmax(0, 1 - label * prediction);
}

// Función para entrenar el modelo SVM usando SGD
void train_svm(DataPoint data[], double weights[], int n_samples, int n_features, KernelType kernel_type) {
    double *gradients = (double *)malloc(n_features * sizeof(double));

    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        // Inicializar los gradientes
        for (int j = 0; j < n_features; j++) {
            gradients[j] = 0.0;
        }

        // Actualizar pesos usando SGD
        for (int i = 0; i < n_samples; i++) {
            double prediction = 0.0;
            if (kernel_type == LINEAR) {
                prediction = dot_product(data[i].features, weights, n_features);
            }
            // Si se desea implementar otros kernels como POLYNOMIAL, añádalos aquí

            double error = hinge_loss(prediction, data[i].label);

            if (error > 0) {
                for (int j = 0; j < n_features; j++) {
                    gradients[j] += (2.0 / n_samples) * (data[i].label - prediction) * data[i].features[j];
                }
            }
        }

        // Actualizar los pesos
        for (int j = 0; j < n_features; j++) {
            weights[j] += LEARNING_RATE * gradients[j] - LEARNING_RATE * C * weights[j];
        }
    }

    free(gradients);
}

// Función para hacer predicciones con el modelo SVM
double predict_svm(double features[], double weights[], int n_features, KernelType kernel_type) {
    if (kernel_type == LINEAR) {
        return dot_product(features, weights, n_features);
    }
    // Si se desea implementar otros kernels como POLYNOMIAL, añádalos aquí
    return 0.0;
}

// Inicialización de la red neuronal
NeuralNetwork initialize_nn(int n_layers, int *neurons_per_layer) {
    NeuralNetwork nn;
    nn.n_layers = n_layers;
    nn.layers = (Layer *)malloc(n_layers * sizeof(Layer));

    for (int i = 0; i < n_layers; ++i) {
        nn.layers[i].n_neurons = neurons_per_layer[i];
        nn.layers[i].z = (double *)malloc(neurons_per_layer[i] * sizeof(double));
        nn.layers[i].a = (double *)malloc(neurons_per_layer[i] * sizeof(double));
        nn.layers[i].b = (double *)malloc(neurons_per_layer[i] * sizeof(double));

        if (i > 0) {
            nn.layers[i].w = (double **)malloc(neurons_per_layer[i] * sizeof(double *));
            for (int j = 0; j < neurons_per_layer[i]; ++j) {
                nn.layers[i].w[j] = (double *)malloc(neurons_per_layer[i - 1] * sizeof(double));
                for (int k = 0; k < neurons_per_layer[i - 1]; ++k) {
                    nn.layers[i].w[j][k] = (double)rand() / RAND_MAX * 2 - 1; // Inicialización aleatoria
                }
                nn.layers[i].b[j] = (double)rand() / RAND_MAX * 2 - 1; // Inicialización aleatoria
            }
        }
    }

    return nn;
}

// Propagación hacia adelante
void forward_propagation(NeuralNetwork *nn, double *input) {
    for (int i = 0; i < nn->layers[0].n_neurons; ++i) {
        nn->layers[0].a[i] = input[i];
    }

    for (int i = 1; i < nn->n_layers; ++i) {
        for (int j = 0; j < nn->layers[i].n_neurons; ++j) {
            double z = nn->layers[i].b[j];
            for (int k = 0; k < nn->layers[i - 1].n_neurons; ++k) {
                z += nn->layers[i].w[j][k] * nn->layers[i - 1].a[k];
            }
            nn->layers[i].z[j] = z;
            nn->layers[i].a[j] = relu(z); // Usar ReLU como función de activación
        }
    }
}

// Retropropagación y actualización de pesos
void back_propagation(NeuralNetwork *nn, double *input, double *output, double learning_rate) {
    forward_propagation(nn, input);

    int n_layers = nn->n_layers;
    double **delta = (double **)malloc(n_layers * sizeof(double *));
    for (int i = 0; i < n_layers; ++i) {
        delta[i] = (double *)malloc(nn->layers[i].n_neurons * sizeof(double));
    }

    // Calcular delta para la capa de salida
    for (int i = 0; i < nn->layers[n_layers - 1].n_neurons; ++i) {
        double a = nn->layers[n_layers - 1].a[i];
        delta[n_layers - 1][i] = (a - output[i]) * relu_prime(nn->layers[n_layers - 1].z[i]);
    }

    // Calcular delta para las capas ocultas
    for (int l = n_layers - 2; l >= 0; --l) {
        for (int i = 0; i < nn->layers[l].n_neurons; ++i) {
            double sum = 0.0;
            for (int j = 0; j < nn->layers[l + 1].n_neurons; ++j) {
                sum += nn->layers[l + 1].w[j][i] * delta[l + 1][j];
            }
            delta[l][i] = sum * relu_prime(nn->layers[l].z[i]);
        }
    }

    // Actualizar pesos y sesgos
    for (int l = 1; l < n_layers; ++l) {
        for (int i = 0; i < nn->layers[l].n_neurons; ++i) {
            nn->layers[l].b[i] -= learning_rate * delta[l][i];
            for (int j = 0; j < nn->layers[l - 1].n_neurons; ++j) {
                nn->layers[l].w[i][j] -= learning_rate * delta[l][i] * nn->layers[l - 1].a[j];
            }
        }
    }

    for (int i = 0; i < n_layers; ++i) {
        free(delta[i]);
    }
    free(delta);
}

// Imprimir los pesos de la red neuronal
void print_weights(NeuralNetwork *nn) {
    for (int l = 1; l < nn->n_layers; ++l) {
        printf("Capa %d:\n", l);
        for (int i = 0; i < nn->layers[l].n_neurons; ++i) {
            printf(" Neurona %d:\n", i);
            for (int j = 0; j < nn->layers[l - 1].n_neurons; ++j) {
                printf("  Peso[%d][%d]: %f\n", i, j, nn->layers[l].w[i][j]);
            }
            printf("  Sesgo[%d]: %f\n", i, nn->layers[l].b[i]);
        }
    }
}
