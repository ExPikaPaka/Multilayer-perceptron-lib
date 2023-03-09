#pragma once

class Layer {
public:
	Layer();
	~Layer();

	int neuronsCount;
	int weightsCount;

	double* neurons;
	double* weights;
	double* biasWeights;
	double bias;

	double* nodeValues;
	double* gradients;
	double* biasGradients;

	bool init(int neuronsCount, int neuronsCountInPreviousLayer);
	void clearNeurons();
	void clearNodes();
	void clearGradients();
	void clearWeights();

	double& operator[] (int index);
};



class NeuralNetwork {
public:
	NeuralNetwork();
	NeuralNetwork(int layersCount, int neuronsInLayerCount[]);
	~NeuralNetwork();

	Layer* layer;
	Layer trainingLayer;

	int layersCount;
	bool outputDebugInfo;

	void forwadPropagation(double inputVector[] = 0);
	void updateAllGradients();
	void applyAllGradients(double learningRate);

	void calculateLayer(Layer& input, Layer& target);
	void updateGradients(Layer& target, Layer& previousLayer);
	void applyGradients(Layer& target, double learningRate);
	void calculateOutputLayerNodeValues();
	void calculateOtherLayersNodeValues(Layer& target, Layer& nextLayer);
	double nodeCost(double& value, double& expectedValue);
	double nodeCostDerivative(double& value, double& expectedValue);
	double totalCost();

	void fillInputVector(double inputVector[]);
	void fillExpectedVector(double expectedVector[]);

	void fillWeights();
	void fillWeights(double x);

	bool loadWeights(char fileName[]);
	bool saveWeights(char fileName[]);

	double activation(double x);
	double activationDerivative(double x);

};

