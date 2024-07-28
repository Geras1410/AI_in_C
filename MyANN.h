#include <stdio.h>
#include <stdlib.h>
#include <math.h>

typedef struct neuron_t {
    float actv;
    float *out_weights;
    float bias;
    float z;
    float dactv;
    float *dw;
    float dbias;
    float dz;
} neuron;

typedef struct layer_t {
    int num_neu;
    struct neuron_t *neu;
} layer;

typedef struct neural_network_t {
    int num_layers;
    struct layer_t *layers;
} neural_network;

neuron create_neuron(int num_out_weights);
layer create_layer(int num_neurons, int num_out_weights);
neural_network create_architecture(int num_layers, int *neurons_per_layer);
float sigmoid(float z);
float relu(float z);
void forward_prop(neural_network *nn, float *input);
void back_prop(neural_network *nn, float *input, float *output, float learning_rate);
void train(neural_network *nn, float *input, float *output, int num_samples, float learning_rate, int epochs);
void predict(neural_network *nn, float *input);

// Implementación de las funciones

//Crear neurona funciona bien
neuron create_neuron(int num_out_weights) {
    neuron n;
    n.actv = 0.0;
    n.z = 0.0;
    n.bias = (float)rand() / RAND_MAX * 2 - 1;  // Inicialización aleatoria del sesgo
    n.out_weights = (float *)malloc(num_out_weights * sizeof(float));
    n.dw = (float *)malloc(num_out_weights * sizeof(float));

    //printf("act[%f], bias[%f], z[%f], dact[%f], dwht[%f], dbias[%f], dz[%f]");

    for (int i = 0; i < num_out_weights; i++) {
        n.out_weights[i] = (float)rand() / RAND_MAX * 2 - 1;  // Inicialización aleatoria de pesos
    }

    return n;
}

//Crear capas funciona bien
layer create_layer(int num_neurons, int num_out_weights) {
    layer l;
    l.num_neu = num_neurons;
    l.neu = (neuron *)malloc(num_neurons * sizeof(neuron));

    for (int i = 0; i < num_neurons; i++) {
        l.neu[i] = create_neuron(num_out_weights);
    }

    return l;
}

//Crear la arquitectura funciona bien
neural_network create_architecture(int num_layers, int *neurons_per_layer) {
    neural_network nn;
    nn.num_layers = num_layers;
    nn.layers = (layer *)malloc(num_layers * sizeof(layer));

    for (int i = 0; i < num_layers; i++) {
        int num_out_weights = (i == num_layers - 1) ? 0 : neurons_per_layer[i + 1];
        nn.layers[i] = create_layer(neurons_per_layer[i], num_out_weights);
    }

    return nn;
}

float sigmoid(float z) {
    return 1.0 / (1.0 + exp(-z));
}

float relu(float z) {
    return z > 0 ? z : 0;
}

void forward_prop(neural_network *nn, float *input) {
    // Establecer las activaciones de la capa de entrada
    for (int i = 0; i < nn->layers[0].num_neu; i++) {
        nn->layers[0].neu[i].actv = input[i];
    }

    // Propagación hacia adelante
    for (int l = 1; l < nn->num_layers; l++) {
        for (int j = 0; j < nn->layers[l].num_neu; j++) {
            neuron *n = &nn->layers[l].neu[j];
            n->z = n->bias;
            n->dz = 0.0;

            for (int i = 0; i < nn->layers[l - 1].num_neu; i++) {
                n->z += nn->layers[l - 1].neu[i].actv * nn->layers[l - 1].neu[i].out_weights[j];
            }

            n->actv = sigmoid(n->z);  // Puedes cambiar la función de activación aquí
        }
    }
}

void back_prop(neural_network *nn, float *input, float *output, float learning_rate) {
    int output_layer = nn->num_layers - 1;

    // Calcular delta para la capa de salida
    for (int i = 0; i < nn->layers[output_layer].num_neu; i++) {
        neuron *n = &nn->layers[output_layer].neu[i];
        float error = n->actv - output[i];
        n->dz = error * n->actv * (1 - n->actv);
        n->dbias += n->dz;
    }

    // Retropropagación
    for (int l = output_layer - 1; l >= 0; l--) {
        for (int i = 0; i < nn->layers[l].num_neu; i++) {
            neuron *n = &nn->layers[l].neu[i];
            n->dz = 0.0;

            for (int j = 0; j < nn->layers[l + 1].num_neu; j++) {
                neuron *n_next = &nn->layers[l + 1].neu[j];
                n->dz += n_next->dz * n->out_weights[j];
                n->dw[j] += n_next->dz * n->actv;
                n->out_weights[j] -= learning_rate * n->dw[j];
            }

            n->dz *= n->actv * (1 - n->actv);
            n->bias -= learning_rate * n->dbias;
        }
    }
}

void train(neural_network *nn, float *input, float *output, int num_samples, float learning_rate, int epochs) {
    for (int e = 0; e < epochs; e++) {
        for (int s = 0; s < num_samples; s++) {
            forward_prop(nn, &input[s * nn->layers[0].num_neu]);
            back_prop(nn, &input[s * nn->layers[0].num_neu], &output[s * nn->layers[nn->num_layers - 1].num_neu], learning_rate);
        }
    }
}

void predict(neural_network *nn, float *input) {
    forward_prop(nn, input);

    int output_layer = nn->num_layers - 1;
    for (int i = 0; i < nn->layers[output_layer].num_neu; i++) {
        printf("Output %d: %f\n", i, nn->layers[output_layer].neu[i].actv);
    }
}
