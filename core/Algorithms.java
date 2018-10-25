package core;

import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.concurrent.ThreadLocalRandom;

/**
 * Contains static methods that implement training and prediction algorithms.
 */
public class Algorithms {

    /**
     * Calculates alpha and beta parameters that maximise the likelihood for the known labels in the network
     * @param network network
     * @param numRounds number of gradient descent / Gibbs sampling rounds
     * @param learningRate learning rate
     * @return alpha and beta parameters
     */
    public static Parameters train(Network network, int numRounds, double learningRate) {
        Parameters parameters = new Parameters(); // initialised to zero
        Assignment targetAssignment = Assignment.samplingUnknownGivenKnown(network);
        Assignment observedAssignment = Assignment.samplingAll(network);

        for (int round = 0; round < numRounds; round++) {
            // Performing one Gibbs sampling round per gradient descent round - this is an arbitrary choice
            targetAssignment.performSamplingRound(parameters); // does nothing in case where all training labels known
            observedAssignment.performSamplingRound(parameters);
            network.performGradientDescentRound(targetAssignment, observedAssignment, parameters, learningRate);
        }

        return parameters;
    }

    /**
     * Estimates marginal probabilities for each unlabelled vertex of having a positive label, based on model with
     * the supplied parameters
     * @param network network
     * @param parameters parameters for model
     * @param numRoundsBurnIn number of Gibbs sampling rounds before first sample
     * @param numRoundsBetweenSamples number of Gibbs sampling rounds between successive samples
     * @param numSamples number of samples to take for computing probability
     * @return map (unlabelled vertex) -> (probability of true label, given model)
     */
    public static Map<Vertex, Double> predict(Network network, Parameters parameters,
                                              int numRoundsBurnIn, int numRoundsBetweenSamples, int numSamples) {
        Assignment assignment = Assignment.samplingUnknownGivenKnown(network);

        // Sampling
        Map<Vertex, Integer> positiveCountsByVertex = new HashMap<>(); // i.e. num +ve labels sampled so far
        for (int sample = 0; sample < numSamples; sample++) {
            int numRounds = (sample == 0) ? numRoundsBurnIn : numRoundsBetweenSamples;
            for (int round = 0; round < numRounds; round++) {
                assignment.performSamplingRound(parameters);
            }
            for (Vertex vertex : assignment.getVerticesToSample()) {
                if (assignment.getLabelFor(vertex)) {
                    positiveCountsByVertex.compute(vertex, (key, value) -> (value == null) ? 1 : value + 1);
                }
            }
        }

        // Computing probabilities from the samples (only done for the unlabelled vertices)
        Map<Vertex, Double> probabilities = new HashMap<>();
        for (Vertex vertex : assignment.getVerticesToSample()) {
            int positiveCount = positiveCountsByVertex.computeIfAbsent(vertex, key -> 0);
            double probability = ((double) positiveCount) / ((double) numSamples);
            probabilities.put(vertex, probability);
        }

        return Collections.unmodifiableMap(probabilities);
    }

    /**
     * Creates a new network with the same structure as the existing network, but with the previously-unlabelled
     * vertices now labelled by Gibbs sampling from a model with the parameters provided. This is useful for setting up
     * test cases.
     * @param network old network
     * @param parameters parameters for model
     * @param numRounds number of Gibbs sampling rounds
     * @return new network with previously-unlabelled vertices now labelled by Gibbs sampling
     */
    public static Network sampleMissingLabels(Network network, Parameters parameters, int numRounds) {
        Assignment assignment = Assignment.samplingUnknownGivenKnown(network);

        // Sampling
        for (int round = 0; round < numRounds; round++) {
            assignment.performSamplingRound(parameters);
        }

        // Grabbing labels - will contain the original labelled for the originally-labelled points, as well as the
        // sampled labels for the originally-unlabelled points
        Map<Vertex, Boolean> newLabels = new HashMap<>();
        for (Vertex vertex : network.getVertices()) {
            newLabels.put(vertex, assignment.getLabelFor(vertex));
        }

        // creating shallow copies, so that adding vertices/edges to the original network won't affect the new network
        return new Network(new HashSet<>(network.getVertices()), new HashSet<>(network.getEdges()), newLabels);
    }

    /**
     * Creates a new network with the same structure as the existing network, but with some of the labels erased.
     * This is useful for setting up test cases.
     * @param network old network
     * @param propLabelsToRetain proportion of old labels to keep
     * @return new network with previously-labelled vertices now erased
     */
    public static Network eraseLabels(Network network, double propLabelsToRetain) {
        Map<Vertex, Boolean>  retainedLabels = new HashMap<>();
        for (Map.Entry<Vertex, Boolean> entry : network.getLabels().entrySet()) {
            if (ThreadLocalRandom.current().nextDouble() < propLabelsToRetain) {
                retainedLabels.put(entry.getKey(), entry.getValue());
            }
        }
        return new Network(new HashSet<>(network.getVertices()), new HashSet<>(network.getEdges()), retainedLabels);
    }

}
