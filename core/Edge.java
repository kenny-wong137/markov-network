package core;

class Edge {

    // NB model uses undirected graph; the distinction between from/to is purely for identifying objects in the code
    private final Vertex fromVertex;
    private final Vertex toVertex;

    Edge(Vertex fromVertex, Vertex toVertex) {
        this.fromVertex = fromVertex;
        this.toVertex = toVertex;
    }

    private double getBetaDerivativeForLabels(boolean fromLabel, boolean toLabel) {
        return BooleanUtils.multiply(fromLabel, toLabel);
    }

    // version for gradient descent - using current assigned labels for both vertices
    double getBetaDerivative(Assignment assignment) {
        return getBetaDerivativeForLabels(assignment.getLabelFor(fromVertex), assignment.getLabelFor(toVertex));
    }

    // next two versions are for Gibbs sampling - using hypothetical candidate label for one vertex
    // while using the current assigned label for the other vertex

    double getBetaDerivativeForFromLabel(boolean fromLabel, Assignment assignment) {
        return getBetaDerivativeForLabels(fromLabel, assignment.getLabelFor(toVertex));
    }

    double getBetaDerivativeForToLabel(boolean toLabel, Assignment assignment) {
        return getBetaDerivativeForLabels(assignment.getLabelFor(fromVertex), toLabel);
    }

}
