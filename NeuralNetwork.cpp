#define _USE_MATH_DEFINES

#include "NeuralNetwork.h"
#include <iostream>
#include <fstream>
#include <cmath>
#include <thread>


//------------------------------------- Layer section -------------------------------------//
Layer::Layer() {
	neuronsCount = 0;
	weightsCount = 0;

	neurons = 0;
	weights = 0;
	biasWeights = 0;

	nodeValues = 0;
	gradients = 0;
	biasGradients = 0;

	bias = 1;
}


Layer::~Layer() {
	if (neurons != 0) delete[] neurons;
	if (weights != 0) delete[] weights;
	if (biasWeights != 0) delete[] biasWeights;
	if (nodeValues != 0) delete[] nodeValues;
	if (gradients != 0) delete[] gradients;
	if (biasGradients != 0) delete[] biasGradients;
}


bool Layer::init(int neuronsCount, int neuronsCountInPreviousLayer) {
	// Initializates specified neurons count, and needed amount of weights
	// dependent on previous layer neurons count
	//
	// note: nodeValues are used for backprop algorithm
	//		 basicly they are nodeCostDerivative * activationDerivative

	this->neuronsCount = neuronsCount;
	weightsCount = neuronsCount * neuronsCountInPreviousLayer;

	neurons = new double[neuronsCount];
	nodeValues = new double[neuronsCount];

	weights = new double[neuronsCount * neuronsCountInPreviousLayer];
	gradients = new double[neuronsCount * neuronsCountInPreviousLayer];

	biasWeights = new double[neuronsCount];
	biasGradients = new double[neuronsCount];

	clearNeurons();
	clearWeights();
	clearNodes();
	clearWeights();
	clearGradients();

	if (neurons != 0) return true;
	return false;
}


void Layer::clearNeurons() {
	for (int neuronIndex = 0; neuronIndex < neuronsCount; neuronIndex++) {
		neurons[neuronIndex] = 0.0;
	}
}


void Layer::clearNodes() {
	for (int nodeIndex = 0; nodeIndex < neuronsCount; nodeIndex++) {
		nodeValues[nodeIndex] = 0.0;
	}
}


void Layer::clearGradients() {
	for (int gradientIndex = 0; gradientIndex < weightsCount; gradientIndex++) {
		gradients[gradientIndex] = 0.0;
	}

	for (int biasGradientIndex = 0; biasGradientIndex < neuronsCount; biasGradientIndex++) {
		biasGradients[biasGradientIndex] = 0.0;
	}
}


void Layer::clearWeights() {
	for (int weightIndex = 0; weightIndex < weightsCount; weightIndex++) {
		weights[weightIndex] = 0.0;
	}

	for (int biasWeightIndex = 0; biasWeightIndex < neuronsCount; biasWeightIndex++) {
		biasWeights[biasWeightIndex] = 0.0;
	}
}


double& Layer::operator[] (int index) {
	return neurons[index];
}
//------------------------------------- ============= -------------------------------------//



//--------------------------------- Neural Network section --------------------------------//
NeuralNetwork::NeuralNetwork() {
	layersCount = 0;
	layer = 0;
	outputDebugInfo = 0;
}


NeuralNetwork::~NeuralNetwork() {
	if (layer != 0) delete[] layer;
}


NeuralNetwork::NeuralNetwork(int layersCount, int neuronsInLayerCount[]) {
	// Pre-initialization
	outputDebugInfo = 0;

	// Initializing layers from given list
	this->layersCount = layersCount;
	layer = new Layer[layersCount];


	layer[0].init(neuronsInLayerCount[0], 0);

	for (int layerIndex = 1; layerIndex < layersCount; layerIndex++) {
		layer[layerIndex].init(neuronsInLayerCount[layerIndex], neuronsInLayerCount[layerIndex - 1]);
	}

	trainingLayer.init(neuronsInLayerCount[layersCount - 1], 0);
}


void NeuralNetwork::forwadPropagation(double inputVector[]) {
	// Setting up input vector
	if (inputVector != 0) {
		fillInputVector(inputVector);
	}

	// Forward propagation trought all layers
	for (int layerIndex = 1; layerIndex < layersCount; layerIndex++) {

		// Clearing target layer 
		layer[layerIndex].clearNeurons();

		// Calculating target layer
		calculateLayer(layer[layerIndex - 1], layer[layerIndex]);
	}

}


void NeuralNetwork::calculateLayer(Layer& input, Layer& target) {
	int weightShift = 0;
	// Multiplying each neuron on previous layer by the corresponding weight, and summing it
	// to calculate one separate neuron on the target layer
	// 
	// weightShift is used to shift index so, that on NN 2-3, w1 w2 correspond for first neuron
	// on the second layer, w3 w4 correspond for the second neuron on second layer, and so on.

	// target loop
	for (int targetIndex = 0; targetIndex < target.neuronsCount; targetIndex++) {

		// input loop
		for (int inputIndex = 0; inputIndex < input.neuronsCount; inputIndex++) {
			target.neurons[targetIndex] += input.neurons[inputIndex] * target.weights[inputIndex + weightShift];
		}
		target.neurons[targetIndex] += input.bias * target.biasWeights[targetIndex];

		// Activating value
		target.neurons[targetIndex] = activation(target.neurons[targetIndex]);

		// Shiffting weights index
		weightShift += input.neuronsCount;
	}
}


void NeuralNetwork::updateGradients(Layer& target, Layer& previousLayer) {
	// Updates weight gradients for specified layer
	// Example:
	// gradient = activateInputFromPreviousLayer * nodeValue

	int weightShift = 0;

	for (int neuronIndex = 0; neuronIndex < target.neuronsCount; neuronIndex++) {
		for (int prevLayNeuronInd = 0; prevLayNeuronInd < previousLayer.neuronsCount; prevLayNeuronInd++) {
			target.gradients[weightShift + prevLayNeuronInd] += target.nodeValues[neuronIndex] * previousLayer.neurons[prevLayNeuronInd];
		}
		weightShift += previousLayer.neuronsCount;

		target.biasGradients[neuronIndex] += target.nodeValues[neuronIndex];
	}
}



void NeuralNetwork::updateAllGradients() {
	// Updates all gradients dependent on training layer
	//
	// Doesn`t update weights. Only gradients
	

	// Output layer gradients update
	layer[layersCount - 1].clearNodes();
	calculateOutputLayerNodeValues();
	updateGradients(layer[layersCount - 1], layer[layersCount - 2]);


	// Other layers gradient update
	// nodeValue update for each layer
	for (int layerIndex = layersCount - 2; layerIndex > 0; layerIndex--) {
		layer[layerIndex].clearNodes();
		calculateOtherLayersNodeValues(layer[layerIndex], layer[layerIndex + 1]);
		updateGradients(layer[layerIndex], layer[layerIndex - 1]);
	}
}


void NeuralNetwork::applyGradients(Layer& target, double learningRate) {
	// Updates weights for specified layer, dependent on gradients
	// that must be calculated before

	for (int weightIndex = 0; weightIndex < target.weightsCount; weightIndex++) {
		target.weights[weightIndex] -= target.gradients[weightIndex] * learningRate;
	}

	for (int biasWeightIndex = 0; biasWeightIndex < target.neuronsCount; biasWeightIndex++) {
		target.biasWeights[biasWeightIndex] -= target.biasGradients[biasWeightIndex] * learningRate;
	}
}


void NeuralNetwork::applyAllGradients(double learningRate) {
	// Going backwards, updates weights for each layer

	for (int layerIndex = layersCount - 1; layerIndex > 0; layerIndex--) {
		applyGradients(layer[layerIndex], learningRate);
	}

	for (int layerIndex = layersCount - 1; layerIndex > 0; layerIndex--) {
		layer[layerIndex].clearGradients();
		layer[layerIndex].clearNodes();
	}
}


void NeuralNetwork::calculateOutputLayerNodeValues() {
	// Calculates nodeValue for each neuron on output layer

	double _costDerivative;
	double _activationDerivative;

	for (int neuronIndex = 0; neuronIndex < layer[layersCount - 1].neuronsCount; neuronIndex++) {
		// Calculating derivatives
		_costDerivative = nodeCostDerivative(layer[layersCount - 1].neurons[neuronIndex], trainingLayer.neurons[neuronIndex]);
		_activationDerivative = activationDerivative(layer[layersCount - 1].neurons[neuronIndex]);

		// Calculating nodeValue
		layer[layersCount - 1].nodeValues[neuronIndex] = _costDerivative * _activationDerivative;
	}
}


void NeuralNetwork::calculateOtherLayersNodeValues(Layer& target, Layer& nextLayer) {
	// Calculates nodeValue for each neuron on target layer, dependent on nexLayer
	// 
	// Example:
	// nodeValue = nextLayer.nodeValue * weightBetweenLayers * activationDerivative


	for (int targetIndex = 0; targetIndex < target.neuronsCount; targetIndex++) {
		for (int nextLayerIndex = 0; nextLayerIndex < nextLayer.neuronsCount; nextLayerIndex++) {
			target.nodeValues[targetIndex] += nextLayer.nodeValues[nextLayerIndex] *
				nextLayer.weights[nextLayerIndex * target.neuronsCount + targetIndex];
		}
		target.nodeValues[targetIndex] *= activationDerivative(target.neurons[targetIndex]);
	}
}


double NeuralNetwork::nodeCost(double& value, double& expectedValue) {
	// Calculates single activated neuron error
	// 
	// error * error is used to make total error big on big values
	// and small on small values. 
	// Examples:
	// error = 0.9 -> return 0.81
	// error = 0.2 -> return 0.04

	double error = value - expectedValue;
	return error * error;
}


double NeuralNetwork::nodeCostDerivative(double& value, double& expectedValue) {
	return 2.0f * (value - expectedValue);
}


double NeuralNetwork::totalCost() {
	// Calculates cost for each neuron, and return square root of their sum

	double totalCost = 0;

	for (int neuronIndex = 0; neuronIndex < layer[layersCount - 1].neuronsCount; neuronIndex++) {
		totalCost += nodeCost(layer[layersCount - 1].neurons[neuronIndex], trainingLayer.neurons[neuronIndex]);
	}

	return sqrt(totalCost);
}


void NeuralNetwork::fillInputVector(double inputVector[]) {
	for (int neuronIndex = 0; neuronIndex < layer[0].neuronsCount; neuronIndex++) {
		layer[0].neurons[neuronIndex] = inputVector[neuronIndex];
	}
}


void NeuralNetwork::fillExpectedVector(double expectedVector[]) {
	for (int neuronIndex = 0; neuronIndex < trainingLayer.neuronsCount; neuronIndex++) {
		trainingLayer[neuronIndex] = expectedVector[neuronIndex];
	}
}


void NeuralNetwork::fillWeights() {
	// Filling weights for each layer
	for (int layerIndex = 1; layerIndex < layersCount; layerIndex++) {

		// Filling neurons weights
		for (int weightIndex = 0; weightIndex < layer[layerIndex].weightsCount; weightIndex++) {
			layer[layerIndex].weights[weightIndex] = (double)rand() / (double)RAND_MAX > 0.5 ? (double)rand() / (double)RAND_MAX
				: -(double)rand() / (double)RAND_MAX;
		}

		// Filling bias weights
		for (int weightIndex = 0; weightIndex < layer[layerIndex].neuronsCount; weightIndex++) {
			layer[layerIndex].biasWeights[weightIndex] = (double)rand() / (double)RAND_MAX > 0.5 ? (double)rand() / (double)RAND_MAX
				: -(double)rand() / (double)RAND_MAX;
		}
	}
}


void NeuralNetwork::fillWeights(double x) {
	// Filling weights for each layer
	for (int layerIndex = 1; layerIndex < layersCount; layerIndex++) {

		// Filling neurons weights
		for (int weightIndex = 0; weightIndex < layer[layerIndex].weightsCount; weightIndex++) {
			layer[layerIndex].weights[weightIndex] = x;
		}

		// Filling bias weights
		for (int weightIndex = 0; weightIndex < layer[layerIndex].neuronsCount; weightIndex++) {
			layer[layerIndex].biasWeights[weightIndex] = x;
		}
	}
}


bool NeuralNetwork::loadWeights(char fileName[]) {
	std::string buffer;
	int loopCounter = 0;

	std::ifstream fin;
	fin.precision(16);

	// Opening
	fin.open(fileName);

	if (!fin.is_open()) {
		if (outputDebugInfo) {
			std::cout << "[ERROR] Can`t open specified weights file :\n        " << fileName << '\n';
		}
		return false;
	}

	// Loading
	for (int layerIndex = 1; layerIndex < layersCount; layerIndex++) {
		// Parsing text, until neurons weights text starts
		buffer.clear();
		loopCounter = 0;

		while (buffer.c_str()[0] != ':') {
			fin >> buffer;
			loopCounter++;

			if (loopCounter > 512) {
				if (outputDebugInfo) {
					std::cout << "[ERROR] Something went wrong during loading weights from file :\n        " << fileName << '\n';
					std::cout << "        Check weights file.\n";
				}
				return false;
			}
		}

		// Reading neurons weights
		for (int weightIndex = 0; weightIndex < layer[layerIndex].weightsCount; weightIndex++) {
			fin >> layer[layerIndex].weights[weightIndex];
		}



		// Parsing text, until bias weights starts
		buffer.clear();
		loopCounter = 0;

		while (buffer.c_str()[0] != ':') {
			fin >> buffer;
			loopCounter++;

			if (loopCounter > 512) {
				if (outputDebugInfo) {
					std::cout << "[ERROR] Something went wrong during loading weights from file :\n        " << fileName << '\n';
					std::cout << "        Check weights file.\n";
				}
				return false;
			}
		}

		// Reading bias weights
		for (int weightIndex = 0; weightIndex < layer[layerIndex].neuronsCount; weightIndex++) {
			fin >> layer[layerIndex].biasWeights[weightIndex];
		}
	}

	// Error check
	if (!fin.good()) {
		if (outputDebugInfo) {
			std::cout << "[ERROR] Something went wrong during loading weights from file :\n        " << fileName << '\n';
			std::cout << "        Check weights file.\n";
		}
		return false;
	}

	if (outputDebugInfo) {
		std::cout << "[DEBUG] Weights loaded succesfully! :\n" << "        " << fileName << '\n';
	}
	return true;
}


bool NeuralNetwork::saveWeights(char fileName[]) {
	std::ofstream fout;
	fout.precision(16);

	// Opening
	fout.open(fileName);

	if (!fout.is_open()) {
		if (outputDebugInfo) {
			std::cout << "[ERROR] Can`t open specified weights file :\n        " << fileName << '\n';
		}
		return false;
	}

	// Saving
	for (int layerIndex = 1; layerIndex < layersCount; layerIndex++) {
		fout << "Layer [" << layerIndex - 1 << " -> " << layerIndex << "] neurons weights : \n";

		for (int weightIndex = 0; weightIndex < layer[layerIndex].weightsCount; weightIndex++) {
			fout << layer[layerIndex].weights[weightIndex] << " ";
		}
		fout << "\n";


		fout << "Layer [" << layerIndex - 1 << " -> " << layerIndex << "] bias weights : \n";

		for (int weightIndex = 0; weightIndex < layer[layerIndex].neuronsCount; weightIndex++) {
			fout << layer[layerIndex].biasWeights[weightIndex] << " ";
		}
		fout << "\n\n";
	}

	// Error check
	if (!fout.good()) {
		if (outputDebugInfo) {
			std::cout << "[ERROR] Something went wrong during saving weights from file :\n        " << fileName << '\n';
			std::cout << "        Check weights file. Error may ocure, if specified file is empty\n";
		}
		return false;
	}

	if (outputDebugInfo) {
		std::cout << "[DEBUG] Weights saved succesfully! :\n" << "        " << fileName << '\n';
	}
	return true;
}


double NeuralNetwork::activation(double x) {
	// Sigmoid function
	return 1.0 / (1.0 + exp(-x));
}


double NeuralNetwork::activationDerivative(double x) {
	// Sigmoid derivative
	// 
	// Remember, that x passed to this function
	// is already activated value with sigmoid function
	return x * (1.0 - x);
}

//------------------------------------- ============= -------------------------------------//