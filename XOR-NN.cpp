#include <iostream>
#include <string>
#include <vector>
#include <windows.h>
#include <cmath>
#include <fstream>
using namespace std;


bool UI(int &train_cycles) {
	float query = 0.0f; string answer; bool to_save = false;
	cout << "		XOR::Training model\nEnter training cycles ->";
	cin >> train_cycles; cout << "Export rsults to_file?\nhmm ? ->";
	cin >> answer;
	if (answer == "Yes" || answer == "yes") { return to_save = true; }
	else { return to_save = false; }
}



void FILE_Export(const int& success_rate, const int& training_cycles, double& Execution_time) {
	ofstream FILE; FILE.open("XOR-NN.txt", ios::out);
	if (!FILE.is_open()) { 
	system("cls"); cerr << "file could not be created \n\n";
	system("pause"); FILE.close(); exit(1);
	}

	FILE << "			XOR::training - MODEL\n\n\n";
	FILE << "Training Cycles = " << training_cycles << endl;
	FILE << "With Success rate of: " << success_rate << "%... and Execution time for: " << Execution_time << "s\n\n";
	FILE << "Version 0.1";

	FILE.close();
}



class NeuralNetwork
{
	struct Synapse
	{
		int Afferent;
		int Efferent;
		float fWeight;
	};

	struct Neuron
	{
		vector<int> vecAfferentSynapse;
		vector<int> vecEfferentSynapse;
		float fSummedInput;
		float fOutput;
		float fError;
		bool bIsInput;
		bool bIsOutput;
	};

	vector<float>& vecInput;
	vector<float>& vecOutput;
	vector<Neuron> vecNodes;
	vector<Synapse> vecSynapses;

public:
	NeuralNetwork(vector<float>& in, vector<float>& out, int nHiddenLayers) : vecInput(in), vecOutput(out)
	{
		
		int nNodeCount = 0;

		vector<int> vecPreviousLayerIDs;
		vector<int> vecNewPreviousLayerIDs;

		// Input Layer
		for (auto n : vecInput)
		{
			Neuron node;
			node.bIsInput = true;
			node.bIsOutput = false;
			vecNodes.push_back(node);
			vecNewPreviousLayerIDs.push_back(vecNodes.size() - 1);
		}

		for (int h = 0; h < nHiddenLayers; h++)
		{
			vecPreviousLayerIDs = vecNewPreviousLayerIDs;
			vecNewPreviousLayerIDs.clear();

			for (auto n : vecInput) 
			{
				// Create New Neuron
				Neuron node;
				node.bIsInput = false;
				node.bIsOutput = false;
				vecNodes.push_back(node);
				vecNewPreviousLayerIDs.push_back(vecNodes.size() - 1);

				// Fully connect it to previous layer
				for (auto p : vecPreviousLayerIDs)
				{
					// Create synapse
					Synapse syn;
					syn.Afferent = p;
					syn.fWeight = (float)rand() / (float)RAND_MAX; //0.0f;
					syn.Efferent = vecNodes.size() - 1;
					vecSynapses.push_back(syn);

					vecNodes[p].vecEfferentSynapse.push_back(vecSynapses.size() - 1);

					vecNodes.back().vecAfferentSynapse.push_back(vecSynapses.size() - 1);
				}
			}
		}


		vecPreviousLayerIDs = vecNewPreviousLayerIDs;
		for (auto n : vecOutput) 
		{
			// Create New Neuron
			Neuron node;
			node.bIsInput = false;
			node.bIsOutput = true;
			vecNodes.push_back(node);
			vecNewPreviousLayerIDs.push_back(vecNodes.size() - 1);

			// Fully connect it to previous layer
			for (auto p : vecPreviousLayerIDs)
			{
				// Create synapse
				Synapse syn;
				syn.Afferent = p;
				syn.fWeight = (float)rand() / (float)RAND_MAX; //0.0f;
				syn.Efferent = vecNodes.size() - 1;
				vecSynapses.push_back(syn);

				// Connect Afferent Node to synapse
				vecNodes[p].vecEfferentSynapse.push_back(vecSynapses.size() - 1);

				// Connect this node to synapse
				vecNodes.back().vecAfferentSynapse.push_back(vecSynapses.size() - 1);
			}
		}
	}



	void PropagateForward()
	{
		int in_count = 0;
		int out_count = 0;
		for (auto& n : vecNodes)
		{
			n.fSummedInput = 0;
			if (n.bIsInput)
			{
				// Node is input, so just set output directly
				n.fOutput = vecInput[in_count];
				in_count++;
			}
			else
			{
				// Accumulate input via weighted synapses
				for (auto s : n.vecAfferentSynapse)
					n.fSummedInput += vecNodes[vecSynapses[s].Afferent].fOutput * vecSynapses[s].fWeight;

				// Activation Function
				n.fOutput = 1.0f / (1.0f + expf(-n.fSummedInput * 2.0f));

				if (n.bIsOutput)
				{
					vecOutput[out_count] = n.fOutput;
					out_count++;
				}
			}
		}
	}



	void PropagateBackwards(vector<float>& vecTrain, float fDelta)
	{
		for (int n = vecNodes.size() - 1; n >= 0; n--)
		{
			if (vecNodes[n].bIsOutput) 
			{
				vecNodes[n].fError = (vecTrain[vecTrain.size() - (vecNodes.size() - n)] - vecNodes[n].fOutput) * (vecNodes[n].fOutput * (1.0f - vecNodes[n].fOutput));
			}
			else
			{
				vecNodes[n].fError = 0.0f;
				for (auto effsyn : vecNodes[n].vecEfferentSynapse)
				{
					float fEfferentNeuronError = vecNodes[vecSynapses[effsyn].Efferent].fError;
					float fEfferentSynapseWeight = vecSynapses[effsyn].fWeight;
					vecNodes[n].fError += (fEfferentSynapseWeight * fEfferentNeuronError) * (vecNodes[n].fOutput * (1.0f - vecNodes[n].fOutput));
				}
			}
		}

		// Update Synaptic Weights
		for (auto& s : vecSynapses)
			s.fWeight += fDelta * vecNodes[s.Efferent].fError * vecNodes[s.Afferent].fOutput;
	}

};



int main()
{
	vector<float> input = { 0,0 }; // 0,0 - default
	vector<float> output = { 0 }; // 0 - default
	vector<float> train = { 0 }; // 0 - default

	float success_rate = 0.0f;
	int training_cycles = 0;
	bool to_save = UI(training_cycles);
	double Execution_time = 0.0f;

	time_t start, finished;
	ios_base::sync_with_stdio(false);

	// HERE WE GO
	time(&start);

	NeuralNetwork nn(input, output, 1);

	// XOR Training
	for (int i = 0; i < training_cycles; i++)
	{
		int nA = rand() % 2;
		int nB = rand() % 2;
		int nY = nA ^ nB;

		input[0] = (float)nA;
		input[1] = (float)nB; 
		train[0] = (float)nY;

		nn.PropagateForward();

		int nO = (int)(output[0] + 0.5f);

		if (nO == nY) { success_rate += 1.0f; }
		else { success_rate -= 1.0f; }

		cout << "A: " << nA << " B: " << nB << " Y: " << nY << " NN: " << nO << " (" << to_string(output[0]) << ") " << (nO == nY ? "PASS" : "FAIL :/") <<endl;
		nn.PropagateBackwards(train, 0.5f);
	}

	time(&finished);
	Execution_time = double(start - finished);

	cout << "\n\n\nRate of success: " << 100 * (success_rate / training_cycles)<<"%...\n\n";
	cout << "Time for Execution: " << Execution_time << "s";
	if (to_save) { FILE_Export(success_rate, training_cycles, Execution_time); }

	return 0;
};
