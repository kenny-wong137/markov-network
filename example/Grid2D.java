package example;

import core.Algorithms;
import core.Network;
import core.Parameters;
import core.Vertex;

import java.util.Map;

public class Grid2D {

    private static Network makeUnlabelledNetwork(int gridSize) {
        Network network = new Network();

        Vertex[][] vertices = new Vertex[gridSize][gridSize];
        for (int row = 0; row < gridSize; row++) {
            for (int col = 0; col < gridSize; col++) {
                vertices[row][col] = network.makeUnlabelledVertexWithRandomFeature();
            }
        }

        for (int row = 0; row < gridSize; row++) {
            for (int col = 0; col < gridSize; col++) {
                network.addEdge(vertices[row][col], vertices[(row + 1) % gridSize][col]);
                network.addEdge(vertices[row][col], vertices[row][(col + 1) % gridSize]);
            }
        }

        return network;
    }

    private static Network makeLabelledNetwork(int gridSize, double alpha, double beta, int numLabellingRounds) {
        Network unlabelledNetwork = makeUnlabelledNetwork(gridSize);
        Parameters expectedParams = new Parameters(alpha, beta);
        return Algorithms.sampleMissingLabels(unlabelledNetwork, expectedParams, numLabellingRounds);
    }

    private static void runTrainingExperiment(Network labelledNetwork, int numTrainingRounds, double learningRate) {
        // Sample labels for network based on parameters
        // Use gradient descent algorithm to fit parameters for the labelled network
        // If this works, we should get something close to the original parameters
        Parameters fittedParams = Algorithms.train(labelledNetwork, numTrainingRounds, learningRate);
        System.out.println(fittedParams);
    }

    private static void runPredictionExperiment(Network labelledNetwork, double propLabelsToRetain,
                                                double alpha, double beta, int numRoundsBurnIn,
                                                int numRoundsBetweenSamples, int numSamples) {
        // Sample labels for network based on parameters
        // Erase some of the labels, and try and predict what these labels are based on the parameters passed.
        // If this works, we should get an accuracy above 0.5.
        Map<Vertex, Boolean> trueLabels = labelledNetwork.getLabels();
        Network partiallyLabelledNetwork = Algorithms.eraseLabels(labelledNetwork, propLabelsToRetain);
        Parameters parameters = new Parameters(alpha, beta);
        Map<Vertex, Double> predictions = Algorithms.predict(partiallyLabelledNetwork,
                parameters, numRoundsBurnIn, numRoundsBetweenSamples, numSamples);

        int allCount = 0;
        int correctCount = 0;
        for (Vertex vertex : predictions.keySet()) {
            allCount++;
            if ((predictions.get(vertex) > 0.5) == trueLabels.get(vertex)) {
                correctCount++;
            }
        }
        System.out.printf("Number of predictions = %d, Number correct = %d%n", allCount, correctCount);
    }

    private static void runBothExperiments(int gridSize, double alpha, double beta, int numLabellingRounds,
                                           int numTrainingRounds, double learningRate, double propLabelsToRetain,
                                           int numRoundsBurnIn, int numRoundsBetweenSamples, int numSamples) {
        Network labelledNetwork = makeLabelledNetwork(gridSize, alpha, beta, numLabellingRounds);
        runTrainingExperiment(labelledNetwork, numTrainingRounds, learningRate);
        runPredictionExperiment(labelledNetwork, propLabelsToRetain, alpha, beta, numRoundsBurnIn,
                numRoundsBetweenSamples, numSamples);
    }

    public static void main(String[] args) {
        runBothExperiments(100, 0.3, 0.5, 1000, 1000, 1.0e-5, 0.2, 1000, 50, 50);
    }

}
