package org.tensorflow.model.sample.mnist;

import java.util.Arrays;
import java.util.Iterator;

import org.tensorflow.Graph;
import org.tensorflow.Operand;
import org.tensorflow.Session;
import org.tensorflow.Shape;
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
    Placeholder<Float> images = tf.placeholder(Float.class, Placeholder.shape(Shape.make(-1, 784)));
    Placeholder<Float> labels = tf.placeholder(Float.class);

    Variable<Float> weights = tf.variable(Shape.make(784, 10), Float.class);
    Assign<Float> weightsInit = tf.assign(weights, tf.zeros(constArray(tf, 784, 10), Float.class));

    Variable<Float> biases = tf.variable(Shape.make(10), Float.class);
    Assign<Float> biasesInit = tf.assign(biases, tf.zeros(constArray(tf, 10), Float.class));

    // Build the graph
    Softmax<Float> softmax = tf.nn.softmax(tf.math.add(tf.linalg.matMul(images,  weights), biases));
    Mean<Float> crossEntropy = tf.math.mean(tf.math.neg(tf.reduceSum(tf.math.mul(labels, tf.math.log(softmax)), constArray(tf, 1))), constArray(tf, 0));

    Gradients gradients = tf.gradients(crossEntropy, Arrays.asList(weights, biases));
    Constant<Float> alpha = tf.constant(LEARNING_RATE);
    ApplyGradientDescent<Float> weightGradientDescent = tf.train.applyGradientDescent(weights, alpha, gradients.dy(0));
    ApplyGradientDescent<Float> biasGradientDescent = tf.train.applyGradientDescent(biases, alpha, gradients.dy(1));
  
    
    Operand<Long> predicted = tf.math.argMax(softmax, tf.constant(1));
    Operand<Long> expected = tf.math.argMax(labels, tf.constant(1));
    Operand<Float> accuracy = tf.math.mean(tf.dtypes.cast(tf.math.equal(predicted, expected), Float.class), constArray(tf, 0));

    try (Session session = new Session(graph)) {

      // Initialize graph variables
      session.runner()
          .addTarget(weightsInit)
          .addTarget(biasesInit)
          .run();

      // Train the graph
      for (Iterator<ImageBatch> batchIter = dataset.trainingBatchIterator(TRAINING_BATCH_SIZE); batchIter.hasNext();) {
        ImageBatch batch = batchIter.next();
        try (Tensor<Float> batchImages = Tensor.create(batch.shape(784), batch.images());
             Tensor<Float> batchLabels = Tensor.create(batch.shape(10), batch.labels())) {
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
      try (Tensor<Float> testImages = Tensor.create(testBatch.shape(784), testBatch.images());
           Tensor<Float> testLabels = Tensor.create(testBatch.shape(10), testBatch.labels());
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
  private static Operand<Integer> constArray(Ops tf, int... i) {
    return tf.constant(i);
  }
}
