#include "NeuralNetwork.h"
#include <iostream> /////////////////
#include <time.h>
#include <cmath>


int randSeed = (int)(time(NULL)%10000); /* Random seed */ ///////////////Test max limit
int randA = 4545; /* Random a value */
int randB = 8334; /* Random b value */
int randM = 8427; /* Random m value */

double ActivationFunction(double z){
    double activation = 1.0/(1+exp(z*(-1))); /* Sigmoid */
    //double activation = tanh(z); /* Hyperbolic tangent */

    return activation;
}

int RandomNumber(int zeroToX)
{
    randSeed = (((randSeed*randA) + randB)%randM);

    return (int)floor((((double)randSeed/(double)randM)*(double)zeroToX));
}

Neuron::Neuron(int lastLayer)
{
    for(int nOW = 0; nOW < lastLayer; nOW++){

        int positivNegativ = RandomNumber(2) == 0 ? -1 : 1;
        int answer = RandomNumber(128)*positivNegativ;
        this->weights.push_back(answer); /* Binary: 1111111 */
    }
}

double Neuron::WeightInput(std::vector< double > inputFromLastLayer){
    double neuronSum = 0;
    for(int neuronIndex = 0; neuronIndex < inputFromLastLayer.size()-1; neuronIndex++){
        neuronSum += (inputFromLastLayer.at(neuronIndex) * this->weights.at(neuronIndex));
    }

    return ActivationFunction(neuronSum);
}


Layer::Layer(int numberOfNeurons, int lastLayer)
{
    for(int nON = 0; nON < numberOfNeurons; nON++){
        neurons.push_back(Neuron(lastLayer+1));
    }

}

std::vector< double > Layer::WeightNeurons(std::vector< double > inputFromLastLayer){

    inputFromLastLayer.push_back(1.0);

    std::vector< double > outputVector;
    for(int neuronIndex = 0; neuronIndex < this->neurons.size(); neuronIndex++){
        outputVector.push_back(this->neurons.at(neuronIndex).WeightInput(inputFromLastLayer));
    }
    outputVector.push_back(1); /* Bias value */
    return outputVector;
}

NeuralNetwork::NeuralNetwork(std::vector< int > networkSize)
{
    for(int nOL = 1; nOL < networkSize.size(); nOL++){
        layers.push_back(Layer(networkSize.at(nOL), networkSize.at(nOL-1))); /* Runs through the rest of the layers */
    }
    fitness = 0; /* Fitness is set to zero from beginning */
}

std::vector< double > NeuralNetwork::GetOutput(std::vector< double > NNInput)
{
    for(int layerIndex = 1; layerIndex < this->layers.size(); layerIndex++){
        NNInput = this->layers.at(layerIndex).WeightNeurons(NNInput);
    }
    std::vector< double > outputVector;
    for(int outputWithoutBias = 0; outputWithoutBias < NNInput.size()-1; outputWithoutBias++){
        outputVector.push_back(NNInput.at(outputWithoutBias));
    }

    return outputVector;
}

void NeuralNetwork::addFitness(double fitnessScore)
{
    this->fitness = fitnessScore;
}
double NeuralNetwork::getFitness()
{
    return this->fitness;
}


NetworkContainer::NetworkContainer(int numberOfNetworks, std::vector< int > networkSize)
{
    for(int networkIndex = 0; networkIndex < numberOfNetworks; networkIndex++){
        this->NeuralNetworks.push_back(NeuralNetwork(networkSize));
    }
}


void NetworkContainer::breed()
{

}
