package core;

import java.util.concurrent.ThreadLocalRandom;

class BooleanUtils {

    static double multiply(boolean bool1, boolean bool2) {
        return (bool1 == bool2) ? 1.0 : -1.0;
    }

    static boolean randBool() {
        return ThreadLocalRandom.current().nextBoolean();
    }

    static boolean sampleBool(double probTrue) {
        return ThreadLocalRandom.current().nextDouble() < probTrue;
    }

}
