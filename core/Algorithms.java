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
     * @param descentSteps number of gradient descent steps
     * @param samplingPassesBurnIn number of Gibbs sampling passes before the first gradient descent round
     * @param samplingPassesBetweenDescents number of Gibbs sampling passes between each gradient descent round
     * @param learningRate learning rate
     * @return alpha and beta parameters
     */
    public static Parameters train(Network network, int descentSteps, int samplingPassesBurnIn,
                                   int samplingPassesBetweenDescents, double learningRate) {
        Parameters parameters = new Parameters(); // initialised to zero
        Assignment targetAssignment = Assignment.samplingUnknownGivenKnown(network);
        Assignment observedAssignment = Assignment.samplingAll(network);

        for (int descentStep = 0; descentStep < descentSteps; descentStep++) {
            int samplingPasses = (descentStep == 0) ? samplingPassesBurnIn : samplingPassesBetweenDescents;
            for (int samplingPass = 0; samplingPass < samplingPasses; samplingPass++) {
                targetAssignment.performSamplingRound(parameters); // does nothing if all training labels known
                observedAssignment.performSamplingRound(parameters);
            }
            network.performGradientDescentRound(targetAssignment, observedAssignment, parameters, learningRate);
        }

        return parameters;
    }

    /**
     * Estimates marginal probabilities for each unlabelled vertex of having a positive label, based on model with
     * the supplied parameters
     * @param network network
     * @param parameters parameters for model
     * @param observations number of observations to take for each label, to estimate probability from
     * @param samplingPassesBurnIn number of Gibbs sampling passes before first sample
     * @param samplingPassesBetweenObservations number of Gibbs sampling passes between successive label observations
     * @return map (unlabelled vertex) -> (probability of true label for this vertex, given model)
     */
    public static Map<Vertex, Double> predict(Network network, Parameters parameters, int observations,
                                              int samplingPassesBurnIn, int samplingPassesBetweenObservations) {
        Assignment assignment = Assignment.samplingUnknownGivenKnown(network);

        // Sampling
        Map<Vertex, Integer> positiveCountsByVertex = new HashMap<>(); // i.e. num +ve labels sampled so far
        for (int observation = 0; observation < observations; observation++) {
            int samplingPasses = (observation == 0) ? samplingPassesBurnIn : samplingPassesBetweenObservations;
            for (int samplingPass = 0; samplingPass < samplingPasses; samplingPass++) {
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
            double probability = ((double) positiveCount) / ((double) observations);
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
