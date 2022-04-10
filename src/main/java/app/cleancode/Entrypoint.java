package app.cleancode;

import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Placeholder;
import org.tensorflow.op.math.Mul;
import org.tensorflow.op.math.Square;
import org.tensorflow.types.TFloat64;

public class Entrypoint {
    public static void main(String[] args) {
        try (Graph graph = new Graph()) {
            Ops ops = Ops.create(graph);
            Placeholder<TFloat64> x = ops.placeholder(TFloat64.class);
            Mul<TFloat64> twoX = ops.math.mul(ops.constant(2.0), x);
            Square<TFloat64> x2 = ops.math.square(twoX);

            try (Session session = new Session(graph)) {
                Tensor result =
                        session.runner().feed(x, TFloat64.scalarOf(2.25)).fetch(x2).run().get(0);
                System.out.println(((TFloat64) result).getDouble());
            }
        }
    }
}
