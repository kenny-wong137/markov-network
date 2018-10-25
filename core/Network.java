package core;

import java.util.*;

/**
 * Model:
 * Prob(y | x) = alpha Sum_i x_i y_i + beta Sum_{i,j} y_i y_j
 * where x_i is the feature value for the ith vertex, y_i is the label for the ith vertex
 * and the sum_{i,j} is over all pairs of vertices i, j that are connected by an edge.
 * The x_i's and y_i's take values of +1 or -1.
 * In the cases we consider, the x_i's are known for all i; however, the y_i's are known only for some of the i's.
 * This is true of both the train set and the test set.
 */
public class Network {

    private final Set<Vertex> vertices;
    private final Set<Edge> edges;
    private final Map<Vertex, Boolean> labels;

    Network(Set<Vertex> vertices, Set<Edge> edges, Map<Vertex, Boolean> labels) {
        this.vertices = vertices;
        this.edges = edges;
        this.labels = labels;
    }

    /**
     * An empty network
     */
    public Network() {
        this(new HashSet<>(), new HashSet<>(), new HashMap<>());
    }

    /**
     * Make an unlabelled vertex
     * @param featureValue feature value for vertex
     * @return vertex
     */
    public Vertex makeUnlabelledVertex(boolean featureValue) {
        Vertex newVertex = new Vertex(featureValue);
        vertices.add(newVertex);
        return newVertex;
    }

    /**
     * Make an unlabelled vertex with a random feature value
     * @return vertex
     */
    public Vertex makeUnlabelledVertexWithRandomFeature() {
        return makeUnlabelledVertex(BooleanUtils.randBool()); // with random feature
    }

    /**
     * Make a labelled vertex
     * @param featureValue feature value for vertex
     * @param label label for vertex
     * @return vertex
     */
    public Vertex makeLabelledVertex(boolean featureValue, boolean label) {
        Vertex newVertex = new Vertex(featureValue);
        vertices.add(newVertex);
        labels.put(newVertex, label);
        return newVertex;
    }

    /**
     * Create edge between two vertices. (The direction of the edge doesn't matter)
     * @param fromVertex vertex
     * @param toVertex vertex
     */
    public void addEdge(Vertex fromVertex, Vertex toVertex) {
        Edge newEdge = new Edge(fromVertex, toVertex);
        edges.add(newEdge);
        fromVertex.addOutEdge(newEdge);
        toVertex.addInEdge(newEdge);
    }

    Set<Vertex> getVertices() {
        return vertices;
    }

    Set<Edge> getEdges() {
        return edges;
    }

    /**
     * Returns labels for vertices whose labels are known
     * @return map (vertex) -> (label)
     */
    public Map<Vertex, Boolean> getLabels() {
        return Collections.unmodifiableMap(labels);
    }

    void performGradientDescentRound(Assignment targetAssignment, Assignment observedAssignment,
                                     Parameters parameters, double learningRate) {
        // Goal: Optimise likelihood L = log Prob(y_known | x; alpha, beta)  (marginalising over y_unknown)
        // Partial derivatives:
        // dL/d(alpha) = sum_i E_(y_unknown | y_known)[x_i y_i| alpha,beta] - sum_i E_(y_all)[x_i y_i| alpha,beta]
        // dL/d(beta) = sum_(i,j) E_(y_unknown| y_known)[y_i y_j| alpha,beta] - sum_{i,j) E_(y_all)[y_i y_j| alpha,beta]
        //    (where the sum_{i, j) is over pairs of vertices joined by an edge)
        vertices.parallelStream()
                .forEach(vertex -> {
                    double targetAlphaDerivative = vertex.getAlphaDerivative(targetAssignment);
                    double observedAlphaDerivative = vertex.getAlphaDerivative(observedAssignment);
                    parameters.incrementAlpha(learningRate * (targetAlphaDerivative - observedAlphaDerivative));
                });

        edges.parallelStream()
                .forEach(edge -> {
                    double targetBetaDerivative = edge.getBetaDerivative(targetAssignment);
                    double observedBetaDerivative = edge.getBetaDerivative(observedAssignment);
                    parameters.incrementBeta(learningRate * (targetBetaDerivative - observedBetaDerivative));
                });

        parameters.makeSnapshot();
    }

}
