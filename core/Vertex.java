package core;

import java.util.HashSet;
import java.util.Set;

/**
 * A vertex in the network
 */
public class Vertex {

    private final boolean featureValue;
    private final Set<Edge> outEdges = new HashSet<>();
    private final Set<Edge> inEdges = new HashSet<>();

    Vertex(boolean featureValue) {
        this.featureValue = featureValue;
    }

    void addOutEdge(Edge edge) {
        outEdges.add(edge);
    }

    void addInEdge(Edge edge) {
        inEdges.add(edge);
    }

    // version for Gibbs sampling - uses hypothetical candidate label
    private double getAlphaDerivativeForLabel(boolean label) {
        return BooleanUtils.multiply(label, featureValue);
    }

    // version for gradient descent - use the label that this vertex is currently assigned to
    double getAlphaDerivative(Assignment assignment) {
        return getAlphaDerivativeForLabel(assignment.getLabelFor(this));
    }

    private double getBoltzmannEnergy(boolean label, Assignment assignment, Parameters parameters) {
        // The value returned will be alpha x_i y_i + beta sum_j y_i y_j
        // ... where i is the current vertex and sum_j is over its neighbours
        // Here, y_i is the hypothetical candidate label for this vertex, whereas the y_j's are based on the
        // current assignments for the neighbours
        double energy = parameters.getAlpha() * getAlphaDerivativeForLabel(label);
        for (Edge edge : outEdges) {
            energy += parameters.getBeta() * edge.getBetaDerivativeForFromLabel(label, assignment);
        }
        for (Edge edge : inEdges) {
            energy += parameters.getBeta() * edge.getBetaDerivativeForToLabel(label, assignment);
        }
        return energy;
    }

    void resample(Assignment assignment, Parameters parameters) {
        // by Gibbs sampling
        double energyIfTrue = getBoltzmannEnergy(true, assignment, parameters);
        double energyIfFalse = getBoltzmannEnergy(false, assignment, parameters);

        double probTrue;
        // separate cases, for numeric stability
        if (energyIfTrue >= energyIfFalse) {
            double expNegEnergyDiff = Math.exp(-(energyIfTrue - energyIfFalse));
            probTrue = 1.0 / (1.0 + expNegEnergyDiff);
        } else {
            double expNegEnergyDiff = Math.exp(-(energyIfFalse - energyIfTrue));
            probTrue = expNegEnergyDiff / (1.0 + expNegEnergyDiff);
        }
        boolean newLabel = BooleanUtils.sampleBool(probTrue);

        assignment.setLabelFor(this, newLabel);
    }

}
