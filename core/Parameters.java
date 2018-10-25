package core;

import java.util.concurrent.atomic.DoubleAdder;

/**
 * Parameters for Markov network
 */
public class Parameters {

    private DoubleAdder alphaAdder = new DoubleAdder(); // for writing to from multiple threads during gradient descent
    private DoubleAdder betaAdder = new DoubleAdder();
    private double alphaSnapshot = 0.0; // for reading from during Gibbs sampling
    private double betaSnapshot = 0.0;
    // NB the parameters are only mutated during training.
    // But once made public outside of the package, they will never be mutated again.

    /**
     * Parameters for network
     * @param alpha alpha
     * @param beta beta
     */
    public Parameters(double alpha, double beta) {
        this.alphaAdder.add(alpha);
        this.betaAdder.add(beta);
        makeSnapshot();
    }

    Parameters() {
        // initialised to zero
    }

    void makeSnapshot() {
        // after gradient descent round, will make snapshot of parameters, which can be efficiently read from
        // in the next Gibbs sampling round (since reading from a DoubleAdder is slow).
        alphaSnapshot = alphaAdder.sum();
        betaSnapshot = betaAdder.sum();
    }

    /**
     * Returns value of alpha
     * @return alpha
     */
    public double getAlpha() {
        return alphaSnapshot;
    }

    /**
     * Returns value of beta
     * @return beta
     */
    public double getBeta() {
        return betaSnapshot;
    }

    void incrementAlpha(double increment) {
        alphaAdder.add(increment);
    }

    void incrementBeta(double increment) {
        betaAdder.add(increment);
    }

    @Override
    public String toString() {
        return String.format("alpha = %.5f, beta = %.5f", alphaSnapshot, betaSnapshot);
    }

}
