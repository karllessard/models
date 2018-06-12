package mnist;

import java.util.Arrays;
import java.util.Iterator;

import org.tensorflow.Graph;
import org.tensorflow.Operand;
import org.tensorflow.Session;
import org.tensorflow.Shape;
import org.tensorflow.Tensors;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Assign;
import org.tensorflow.op.core.Constant;
import org.tensorflow.op.core.Gradients;
import org.tensorflow.op.core.Placeholder;
import org.tensorflow.op.core.ReduceMean;
import org.tensorflow.op.core.Softmax;
import org.tensorflow.op.core.Variable;

import mnist.data.ImageBatch;
import mnist.data.ImageDataset;
import util.ExecutionContext;
import util.ExecutionStep;

public class SimpleMnist implements Runnable {

  public static void main(String[] args) {
    try (Graph graph = new Graph()) {
      SimpleMnist mnist = new SimpleMnist(graph);
      mnist.run();
    }
  }

  @Override
  public void run() {
    ImageDataset dataset = ImageDataset.create(VALIDATION_SIZE);

    ExecutionContext initialization = new ExecutionContext();
    ExecutionContext training = new ExecutionContext();
    ExecutionContext testing = new ExecutionContext();
    
    // Create placeholders and variables
    Placeholder<Float> images = tf.placeholder(Float.class, Placeholder.shape(Shape.make(-1, 784)));
    Placeholder<Float> labels = tf.placeholder(Float.class);

    Variable<Float> weights = tf.variable(Shape.make(784, 10), Float.class);
    Assign<Float> weightsInit = tf.assign(weights, tf.zeros(tf.constant(new long[] {784, 10}), Float.class));
    initialization.addTarget(weightsInit);

    Variable<Float> biases = tf.variable(Shape.make(10), Float.class);
    Assign<Float> biasesInit = tf.assign(biases, tf.zeros(tf.constant(new long[] {10L}), Float.class));
    initialization.addTarget(biasesInit);

    // Build the graph
    Softmax<Float> softmax = tf.softmax(tf.add(tf.matMul(images,  weights), biases));
    ReduceMean<Float> crossEntropy = tf.reduceMean(tf.negate(tf.reduceSum(tf.mul(labels, tf.log(softmax)), axes(1))), axes(0));

    Gradients gradients = tf.gradients(crossEntropy, Arrays.asList(weights, biases));
    Constant<Float> alpha = tf.constant(LEARNING_RATE);
    training.addTarget(tf.applyGradientDescent(weights, alpha, gradients.dy(0)));
    training.addTarget(tf.applyGradientDescent(biases, alpha, gradients.dy(1)));

    Operand<Long> predicted = tf.argMax(softmax, axis(1));
    Operand<Long> expected = tf.argMax(labels, axis(1));
    Operand<Float> accuracy = tf.mean(tf.cast(tf.equal(predicted, expected), Float.class), axes(0));
    testing.addTargetToFetch(accuracy);

    try (Session session = new Session(graph)) {

      // Initialize graph variables
      try (ExecutionStep initializationStep = initialization.createStep(session)) {
        initializationStep.run();
      }

      // Train the graph
      for (Iterator<ImageBatch> batchIter = dataset.trainingBatchIterator(TRAINING_BATCH_SIZE); batchIter.hasNext();) {
        try (ExecutionStep trainingStep = training.createStep(session)) {
          ImageBatch batch = batchIter.next();
          trainingStep
            .feed(images, Tensors.create(batch.images()))
            .feed(labels, Tensors.create(batch.labels()))
            .run();
        }
      }

      // Test the graph
      ImageBatch testBatch = dataset.testBatch();
      try (ExecutionStep testingStep = testing.createStep(session)) {
        testingStep
            .feed(images, Tensors.create(testBatch.images()))
            .feed(labels, Tensors.create(testBatch.labels()))
            .run();

        System.out.println("Accuracy: " + testingStep.target(accuracy).floatValue());
      }
    }
  }
  
  private static final int VALIDATION_SIZE = 0;
  private static final int TRAINING_BATCH_SIZE = 100;
  private static final float LEARNING_RATE = 0.5f;
  
  private Graph graph;
  private Ops tf;
  
  private SimpleMnist(Graph graph) {
    this.graph = graph;
    tf = Ops.create(graph);
  }
  
  private Operand<Integer> axis(int i) {
    return tf.constant(1);
  }

  private Operand<Integer> axes(int... i) {
    return tf.constant(i);
  }
}
