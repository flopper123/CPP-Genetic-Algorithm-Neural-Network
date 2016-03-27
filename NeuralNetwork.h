#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include <vector>
#include <string>

int RandomNumber(int zeroToX);

class Neuron
{
    public:
        Neuron(int lastLayer);
        double WeightInput(std::vector< double > inputFromLastLayer);

    private:
        std::vector< int > weights;
        double sum;
};

class Layer
{
    public:
        Layer(int numberOfNeurons, int lastLayer);
        std::vector< double > WeightNeurons(std::vector< double > inputFromLastLayer);
    private:
        std::vector< Neuron > neurons;
};

class NeuralNetwork
{
    public:
        NeuralNetwork(std::vector< int > networkSize);
        std::vector< double > GetOutput(std::vector< double > NNInput);
        void addFitness(double fitnessScore);
        double getFitness();
    private:
        std::vector< Layer > layers;
        double fitness;

};

class NetworkContainer
{
    public:
        NetworkContainer(int NumberOfNetworks, std::vector< int > networkSize);
        void breed();
        //std::vector< std::string > GetNeuralNetworks;
        std::vector< NeuralNetwork > NeuralNetworks;
};

#endif // NEURALNETWORK_H
