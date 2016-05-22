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
        void setWeight(int weight, unsigned int index);
        unsigned int weightsSize();
        int getWeight(unsigned int index);
        std::string GetGenomePart(bool placeSeperator);
    private:
        std::vector< int > weights;
        double sum;
};

class Layer
{
    public:
        Layer(int numberOfNeurons, int lastLayer);
        std::vector< double > WeightNeurons(std::vector< double > inputFromLastLayer);

        void setNeuron(Neuron newNeuron, unsigned int neuronNumber);

        unsigned int neuronsSize();

        Neuron getNeuron(unsigned int neuronIndex);

        std::string GetGenomePart(bool placeSeperator);

    private:
        std::vector< Neuron > neurons;
};

class NeuralNetwork
{
    public:
        NeuralNetwork(std::vector< int > networkSize);

        std::vector< double > GetOutput(std::vector< double > NNInput);

        void addFitness(double fitnessScore);

        double getFitness() const;

        void setLayer(Layer newLayer, unsigned int layerNumber);

        unsigned int layersSize();

        void breed(NeuralNetwork *parent2);

        std::string NetworkToString();

        Layer getLayer(unsigned int LayerIndex);

    private:
        std::vector< Layer > layers;
        double fitness;
        unsigned int totalWeights;
};

class NetworkContainer
{
    public:
        NetworkContainer(unsigned int NumberOfNetworks, std::vector< int > networkSize);
        void breed();
        std::vector< NeuralNetwork > NeuralNetworks;
        double getBestFitness();
        void save(char* filePath);
    private:
        std::vector< int > networksSize;
        double bestFitness;
        std::string bestGenome;
};

#endif // NEURALNETWORK_H
