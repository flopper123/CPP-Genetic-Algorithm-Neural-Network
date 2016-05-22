#include "NeuralNetwork.h"
#include <iostream> /////////////////
#include <time.h>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <string>

int randSeed = (int)(time(NULL)%10000); /* Random seed */ ///////////////Test max limit
int randA = 4545; /* Random a value */
int randB = 8334; /* Random b value */
int randM = 8427; /* Random m value */

double ActivationFunction(double z){
    double activation = 1.0/(1+exp(z*(-1))); /* Sigmoid */
    //double activation = tanh(z); /* Hyperbolic tangent */

    return activation;
}


struct less_than_fitness { /* Used for sorting function */
    inline bool operator() (const NeuralNetwork& NN1, const NeuralNetwork& NN2)
    {
        double fitness1 = NN1.getFitness();
        double fitness2 = NN2.getFitness();

        return (fitness1 > fitness2);
    }
};


int RandomNumber(int zeroToX)
{
    randSeed = (((randSeed*randA) + randB)%randM);
    int answer = (int)floor((((double)randSeed/(double)randM)*(double)zeroToX));
    if(answer != zeroToX){
        return answer;
    } else {
        return answer-1;
    }
}

Neuron::Neuron(int lastLayer)
{
    for(int nOW = 0; nOW < lastLayer; nOW++){

        int positivNegativ = RandomNumber(2) == 0 ? -1 : 1;
        int answer = RandomNumber(128)*positivNegativ;
        this->weights.push_back(answer); /* Binary: 1111111 */
    }
}

void Neuron::setWeight(int weight, unsigned weightIndex)
{
    this->weights.at(weightIndex) = weight;
}

int Neuron::getWeight(unsigned int weightIndex)
{
    return this->weights.at(weightIndex);
}

double Neuron::WeightInput(std::vector< double > inputFromLastLayer){
    double neuronSum = 0;
    for(unsigned int neuronIndex = 0; neuronIndex < inputFromLastLayer.size()-1; neuronIndex++){
        neuronSum += (inputFromLastLayer.at(neuronIndex) * this->weights.at(neuronIndex));
    }

    return ActivationFunction(neuronSum);
}

unsigned int Neuron::weightsSize()
{
    return this->weights.size();
}

std::string base10base2(int number){ /* Reeeally nasty function */
    if(number == 0){
        return "0";
    }
    if(number == 1){
        return "1";
    }
    if(number % 2 == 0){
        return base10base2(number / 2) + "0";
    } else {
        return base10base2(number / 2) + "1";
    }
}


std::string Neuron::GetGenomePart(bool placeSeperator)
{
    std::string theGenes = "";
    for(unsigned int weightIndex = 0; weightIndex < this->weightsSize(); weightIndex++){
        /* Start parsing */
        if(this->weights.at(weightIndex) < 0){
            theGenes += "0"; /* Negative */
        } else {
            theGenes += "1"; /* Positive */
        }
        theGenes += ".";
        std::string base2 = base10base2(abs(this->weights.at(weightIndex)));
        for(int index = 0; index < 7-base2.size(); index++){
            theGenes += "0";
        }
        theGenes += base2;

        bool lastTest = (weightIndex+1 == this->weightsSize() ? true : false);
        if(lastTest == false){
            theGenes += ":";
        }
    }
    if(placeSeperator == false){
        theGenes += "-";
    }
    return theGenes;
}

Layer::Layer(int numberOfNeurons, int lastLayer)
{
    for(int nON = 0; nON < numberOfNeurons; nON++){
        this->neurons.push_back(Neuron(lastLayer+1));
    }

}

std::vector< double > Layer::WeightNeurons(std::vector< double > inputFromLastLayer){

    inputFromLastLayer.push_back(1.0);

    std::vector< double > outputVector;
    for(unsigned int neuronIndex = 0; neuronIndex < this->neurons.size(); neuronIndex++){
        outputVector.push_back(this->neurons.at(neuronIndex).WeightInput(inputFromLastLayer));
    }
    outputVector.push_back(1); /* Bias value */
    return outputVector;
}

void Layer::setNeuron(Neuron newNeuron, unsigned int neuronNumber)
{
    this->neurons.at(neuronNumber) = newNeuron;
}

unsigned int Layer::neuronsSize()
{
    return this->neurons.size();
}

Neuron Layer::getNeuron(unsigned int neuronIndex){
    return this->neurons.at(neuronIndex);
}

std::string Layer::GetGenomePart(bool placeSeperator)
{
    std::string genomePart = "";
    for(unsigned int neuronIndex = 0; neuronIndex < this->neurons.size(); neuronIndex++){
        bool lastTest = (neuronIndex+1 == this->neurons.size() ? true : false);
        genomePart += this->neurons.at(neuronIndex).GetGenomePart(lastTest);
    }
    if(placeSeperator == false){
        genomePart += "#";
    }
    return genomePart;
}



NeuralNetwork::NeuralNetwork(std::vector< int > networkSize)
{
    this->totalWeights = 0;
    for(unsigned int nOL = 1; nOL < networkSize.size(); nOL++){
        this->totalWeights += networkSize.at(nOL-1)*networkSize.at(nOL);
        this->layers.push_back(Layer(networkSize.at(nOL), networkSize.at(nOL-1))); /* Runs through the rest of the layers */
    }
    this->fitness = 0; /* Fitness is set to zero from beginning */
}

std::vector< double > NeuralNetwork::GetOutput(std::vector< double > NNInput)
{
    for(unsigned int layerIndex = 0; layerIndex < this->layers.size(); layerIndex++){
        NNInput = this->layers.at(layerIndex).WeightNeurons(NNInput);
    }
    std::vector< double > outputVector;
    for(unsigned int outputWithoutBias = 0; outputWithoutBias < NNInput.size()-1; outputWithoutBias++){
        outputVector.push_back(NNInput.at(outputWithoutBias));
    }

    return outputVector;
}

void NeuralNetwork::addFitness(double fitnessScore)
{
    this->fitness = fitnessScore;
}
double NeuralNetwork::getFitness() const
{
    return this->fitness;
}

void NeuralNetwork::setLayer(Layer settingLayer, unsigned int layerNumber)
{
    this->layers.at(layerNumber) = settingLayer;
}


unsigned int NeuralNetwork::layersSize()
{
    return this->layers.size();
}

void NeuralNetwork::breed(NeuralNetwork *parent2) /* The breeder */
{
    unsigned int layerNum = RandomNumber(parent2->layersSize()); /* Find layer, in which that you want to split */

    for(unsigned int layerNumIndex = 0; layerNumIndex < layerNum; layerNumIndex++){ /* Run through layers until wanted layer, and copy it in to the other NN */
        this->setLayer(parent2->layers.at(layerNumIndex), layerNumIndex);
    }
    unsigned int neuronNum = RandomNumber(parent2->layers.at(layerNum).neuronsSize()); /* Find splitting Neuron */

    for(unsigned int neuronNumberIndex = 0; neuronNumberIndex < neuronNum; neuronNumberIndex++){ /* Do the same as the layer, but just with neurons */
        this->layers.at(layerNum).setNeuron(parent2->layers.at(layerNum).getNeuron(neuronNumberIndex), neuronNumberIndex);
    }
    unsigned int weightNum = RandomNumber(parent2->layers.at(layerNum).getNeuron(neuronNum).weightsSize());

    for(unsigned int weightNumberIndex = 0; weightNumberIndex < weightNum; weightNumberIndex++){
        this->layers.at(layerNum).getNeuron(neuronNum).setWeight(parent2->layers.at(layerNum).getNeuron(neuronNum).getWeight(weightNumberIndex), weightNumberIndex);
    }
}

std::string NeuralNetwork::NetworkToString()
{
    /**
     *
     * . index value
     * : between each gene
     * - between each neuron
     * # between each layer
     *
     **/
    std::string genome = "";

    for(unsigned int layerIndex = 0; layerIndex < this->layersSize(); layerIndex++){
        bool lastTest = (layerIndex+1 == this->layersSize() ? true : false);
        genome += this->layers.at(layerIndex).GetGenomePart(lastTest);
    }
    return genome;
}

NetworkContainer::NetworkContainer(unsigned int numberOfNetworks, std::vector< int > networkSize)
{
    this->networksSize = networkSize;

    for(unsigned int networkIndex = 0; networkIndex < numberOfNetworks; networkIndex++){
        this->NeuralNetworks.push_back(NeuralNetwork(networkSize));
    }

    this->bestFitness = 0;
}


void NetworkContainer::breed()
{
    /* Sort neural networks */
    std::sort(this->NeuralNetworks.begin(), this->NeuralNetworks.end(), less_than_fitness());

    if(this->NeuralNetworks.at(0).getFitness() > this->bestFitness){
        this->bestFitness = this->NeuralNetworks.at(0).getFitness();
        this->bestGenome = this->NeuralNetworks.at(0).NetworkToString();
    }

    /* Make the acctual breeding */
    for(unsigned int networkIndex = 0; networkIndex < this->NeuralNetworks.size(); networkIndex += 2){
        NeuralNetwork parent1 = this->NeuralNetworks.at(networkIndex);
        this->NeuralNetworks.at(networkIndex).breed(&this->NeuralNetworks.at(networkIndex+1));
        this->NeuralNetworks.at(networkIndex+1).breed(&parent1);
    }

}

double NetworkContainer::getBestFitness(){
    return this->bestFitness;
}

void NetworkContainer::save(char* filePath)
{
    /* Sort neural networks */
    std::sort(this->NeuralNetworks.begin(), this->NeuralNetworks.end(), less_than_fitness());

    std::ofstream NetworkFile;
    NetworkFile.open(filePath);

    NetworkFile << this->bestGenome << "\n";
    NetworkFile << "\n";

    for(unsigned int NetworkIndex = 0; NetworkIndex < this->NeuralNetworks.size(); NetworkIndex++){
        NetworkFile << this->NeuralNetworks.at(NetworkIndex).NetworkToString() + "\n";
    }
    NetworkFile.close();
}
