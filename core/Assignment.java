package core;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.atomic.AtomicBoolean;

class Assignment {

    // Vertices are divided between those whose labels will be kept fixed,
    // ... and those whose labels will be sampled using Gibbs sampling.
    private final Set<Vertex> verticesToSample; // the labels for these vertices will be Gibbs sampled
    private final Map<Vertex, AtomicBoolean> labels; // contains labels for both types of vertices
        // use of AtomicBoolean is purely for memory visibility between threads - no synchronization is used

    private Assignment(Set<Vertex> allVertices, Map<Vertex, Boolean> fixedLabels) {
        verticesToSample = new HashSet<>();
        labels = new HashMap<>();
        for (Vertex vertex : allVertices) {
            if (fixedLabels.containsKey(vertex)) {
                // label fixed - initialise with fixed label
                labels.put(vertex, new AtomicBoolean(fixedLabels.get(vertex)));
            } else {
                // label not fixed - initialise with random label, and put vertex up for sampling
                labels.put(vertex, new AtomicBoolean(BooleanUtils.randBool()));
                verticesToSample.add(vertex);
            }
        }
    }

    // to Gibbs-sample the unknown labels, conditioning on the known labels
    static Assignment samplingUnknownGivenKnown(Network network) {
        return new Assignment(network.getVertices(), network.getLabels());
    }

    // to Gibbs-sample all the labels, conditioning on nothing
    static Assignment samplingAll(Network network) {
        return new Assignment(network.getVertices(), new HashMap<>());
    }

    boolean getLabelFor(Vertex vertex) {
        return labels.get(vertex).get();
    }

    void setLabelFor(Vertex vertex, boolean newLabel) {
        labels.get(vertex).set(newLabel);
    }

    Set<Vertex> getVerticesToSample() {
        return verticesToSample;
    }

    void performSamplingRound(Parameters parameters) {
        // Running in parallel - each vertex will get a view of the other vertices' labels that is up to date
        // either as of this iteration or as of the previous iteration - this should be accurate enough
        verticesToSample.parallelStream()
                .forEach(vertex -> vertex.resample(this, parameters));
    }

}
