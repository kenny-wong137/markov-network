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

    private static void runTrainingExperiment(int gridSize, double alpha, double beta, int numLabellingRounds,
                                              double propLabelsToRetain, int descentSteps, int samplingPassesBurnIn,
                                              int samplingPassesBetweenDescents, double learningRate) {
        // Sample labels for network based on parameters. Keep only a proportion of these for training.
        Network trainingNetwork = Algorithms.eraseLabels(
                makeLabelledNetwork(gridSize, alpha, beta, numLabellingRounds), propLabelsToRetain);

        // Use gradient descent algorithm to fit parameters for the labelled network
        // If this works, we should get something close to the original parameters
        Parameters fittedParams = Algorithms.train(
                trainingNetwork, descentSteps, samplingPassesBurnIn, samplingPassesBetweenDescents, learningRate);
        System.out.println(fittedParams);
    }

    private static void runPredictionExperiment(int gridSize, double alpha, double beta, int numLabellingRounds,
                                                double propLabelsToRetain, int observations, int samplingPassesBurnIn,
                                                int samplingPassesBetweenObservations) {
        // Sample labels for network based on parameters.
        // Only reveal a subset of these labels - it is the job of the model to predict the remaining labels.
        Network fullyLabelledNetwork = makeLabelledNetwork(gridSize, alpha, beta, numLabellingRounds);
        Map<Vertex, Boolean> realLabels = fullyLabelledNetwork.getLabels();
        Network testNetwork = Algorithms.eraseLabels(fullyLabelledNetwork, propLabelsToRetain);

        // Erase some of the labels, and try and predict what these labels are based on the parameters passed.
        // If this works, we should get an accuracy above 0.5.
        Parameters parameters = new Parameters(alpha, beta);
        Map<Vertex, Double> predictions = Algorithms.predict(
                testNetwork, parameters, observations, samplingPassesBurnIn, samplingPassesBetweenObservations);

        int allCount = 0;
        int correctCount = 0;
        for (Vertex vertex : predictions.keySet()) {
            allCount++;
            if ((predictions.get(vertex) > 0.5) == realLabels.get(vertex)) {
                correctCount++;
            }
        }
        System.out.printf("Number of predictions = %d, Number correct = %d%n", allCount, correctCount);
    }

    public static void main(String[] args) {
        // For both experiments
        int gridSize = 100;
        double alpha = 0.3;
        double beta = 0.5;
        int numLabellingRounds = 1000;
        double propLabelsToRetain = 0.5;

        // For training experiment
        int descentSteps = 1000;
        int samplingPassesBurnInForTrain = 250;
        int samplingPassesBetweenDescents = 1;
        double learningRate = 1.0e-5;

        // For prediction experiment
        int observations = 50;
        int samplingPassesBurnInForPredict = 1000;
        int samplingPassesBetweenObservations = 50;

        runTrainingExperiment(gridSize, alpha, beta, numLabellingRounds, propLabelsToRetain,
                descentSteps, samplingPassesBurnInForTrain, samplingPassesBetweenDescents, learningRate);
        runPredictionExperiment(gridSize, alpha, beta, numLabellingRounds, propLabelsToRetain,
                observations, samplingPassesBurnInForPredict, samplingPassesBetweenObservations);
    }

}
