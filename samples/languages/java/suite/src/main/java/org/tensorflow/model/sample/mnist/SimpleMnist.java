package org.tensorflow.model.sample.mnist;

import java.util.Arrays;
import java.util.Iterator;
import org.tensorflow.Graph;
import org.tensorflow.Operand;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.model.sample.mnist.data.ImageBatch;
import org.tensorflow.model.sample.mnist.data.ImageDataset;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Assign;
import org.tensorflow.op.core.Constant;
import org.tensorflow.op.core.Gradients;
import org.tensorflow.op.core.Placeholder;
import org.tensorflow.op.core.Variable;
import org.tensorflow.op.math.Mean;
import org.tensorflow.op.nn.Softmax;
import org.tensorflow.op.train.ApplyGradientDescent;
import org.tensorflow.types.TFloat32;
import org.tensorflow.types.TInt32;
import org.tensorflow.types.TInt64;
import org.tensorflow.tools.Shape;

public class SimpleMnist implements Runnable {

  public static void main(String[] args) {
    ImageDataset dataset = ImageDataset.create(VALIDATION_SIZE);
    try (Graph graph = new Graph()) {
      SimpleMnist mnist = new SimpleMnist(graph, dataset);
      mnist.run();
    }
  }

  @Override
  public void run() {
    Ops tf = Ops.create(graph);
    
    // Create placeholders and variables
    Placeholder<TFloat32> images = tf.placeholder(
        TFloat32.DTYPE, Placeholder.shape(Shape.make(-1, 784)));
    Placeholder<TFloat32> labels = tf.placeholder(TFloat32.DTYPE);

    Variable<TFloat32> weights = tf.variable(Shape.make(784, 10), TFloat32.DTYPE);
    Assign<TFloat32> weightsInit = tf.assign(weights, tf.zeros(constArray(tf, 784, 10), TFloat32.DTYPE));

    Variable<TFloat32> biases = tf.variable(Shape.make(10), TFloat32.DTYPE);
    Assign<TFloat32> biasesInit = tf.assign(biases, tf.zeros(constArray(tf, 10), TFloat32.DTYPE));

    // Build the graph
    Softmax<TFloat32> softmax = tf.nn.softmax(tf.math.add(tf.linalg.matMul(images,  weights), biases));
    Mean<TFloat32> crossEntropy = tf.math.mean(tf.math.neg(tf.reduceSum(tf.math.mul(labels, tf.math.log(softmax)), constArray(tf, 1))), constArray(tf, 0));

    Gradients gradients = tf.gradients(crossEntropy, Arrays.asList(weights, biases));
    Constant<TFloat32> alpha = tf.constant(LEARNING_RATE);
    ApplyGradientDescent<TFloat32> weightGradientDescent = tf.train.applyGradientDescent(weights, alpha, gradients.dy(0));
    ApplyGradientDescent<TFloat32> biasGradientDescent = tf.train.applyGradientDescent(biases, alpha, gradients.dy(1));
  
    Operand<TInt64> predicted = tf.math.argMax(softmax, tf.constant(1));
    Operand<TInt64> expected = tf.math.argMax(labels, tf.constant(1));
    Operand<TFloat32> accuracy = tf.math.mean(tf.dtypes.cast(tf.math.equal(predicted, expected), TFloat32.DTYPE), constArray(tf, 0));

    try (Session session = new Session(graph)) {

      // Initialize graph variables
      session.runner()
          .addTarget(weightsInit)
          .addTarget(biasesInit)
          .run();

      // Train the graph
      for (Iterator<ImageBatch> batchIter = dataset.trainingBatchIterator(TRAINING_BATCH_SIZE); batchIter.hasNext();) {
        ImageBatch batch = batchIter.next();
        try (Tensor<TFloat32> batchImages = Tensor.create(batch.shape(784), batch.images());
             Tensor<TFloat32> batchLabels = Tensor.create(batch.shape(10), batch.labels())) {
          session.runner()
              .addTarget(weightGradientDescent)
              .addTarget(biasGradientDescent)
              .feed(images.asOutput(), batchImages)
              .feed(labels.asOutput(), batchLabels)
              .run();
        }
      }

      // Test the graph
      ImageBatch testBatch = dataset.testBatch();
      try (Tensor<TFloat32> testImages = Tensor.create(testBatch.shape(784), testBatch.images());
           Tensor<TFloat32> testLabels = Tensor.create(testBatch.shape(10), testBatch.labels());
           Tensor<?> value = session.runner()
              .fetch(accuracy)
              .feed(images.asOutput(), testImages)
              .feed(labels.asOutput(), testLabels)
              .run()
              .get(0)) {
        System.out.println("Accuracy: " + value.floatValue());
      }
    }
  }
  
  private static final int VALIDATION_SIZE = 0;
  private static final int TRAINING_BATCH_SIZE = 100;
  private static final float LEARNING_RATE = 0.2f;
  
  private Graph graph;
  private ImageDataset dataset;
  
  private SimpleMnist(Graph graph, ImageDataset dataset) {
    this.graph = graph;
    this.dataset = dataset;
  }

  // Helper that converts a single integer into an array
  private static Operand<TInt32> constArray(Ops tf, int... i) {
    return tf.constant(i);
  }
}
