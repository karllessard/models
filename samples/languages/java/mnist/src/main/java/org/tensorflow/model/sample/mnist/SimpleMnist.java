package org.tensorflow.model.sample.mnist;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;

import org.tensorflow.Graph;
import org.tensorflow.Operand;
import org.tensorflow.Session;
import org.tensorflow.Shape;
import org.tensorflow.Tensor;
import org.tensorflow.Tensors;
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

public class SimpleMnist implements Runnable {

  public static void main(String[] args) {
    try (Graph graph = new Graph()) {
      SimpleMnist mnist = new SimpleMnist(graph);
      mnist.run();
    }
  }

  @Override
  public void run() {
    Ops tf = Ops.create(graph);

    ImageDataset dataset = ImageDataset.create(VALIDATION_SIZE);
    List<Operand<?>> initializationTargets = new ArrayList<>();
    List<Operand<?>> trainingTargets = new ArrayList<>();
    
    // Create placeholders and variables
    Placeholder<Float> images = tf.placeholder(Float.class, Placeholder.shape(Shape.make(-1, 784)));
    Placeholder<Float> labels = tf.placeholder(Float.class);

    Variable<Float> weights = tf.variable(Shape.make(784, 10), Float.class);
    Assign<Float> weightsInit = tf.assign(weights, tf.zeros(tf.constant(new long[] {784, 10}), Float.class));
    initializationTargets.add(weightsInit);

    Variable<Float> biases = tf.variable(Shape.make(10), Float.class);
    Assign<Float> biasesInit = tf.assign(biases, tf.zeros(tf.constant(new long[] {10L}), Float.class));
    initializationTargets.add(biasesInit);

    // Build the graph
    Softmax<Float> softmax = tf.nn.softmax(tf.math.add(tf.linalg.matMul(images,  weights), biases));
    Mean<Float> crossEntropy = tf.math.mean(tf.math.neg(tf.reduceSum(tf.math.mul(labels, tf.math.log(softmax)), constants(tf, 1))), constants(tf, 0));

    Gradients gradients = tf.gradients(crossEntropy, Arrays.asList(weights, biases));
    Constant<Float> alpha = tf.constant(LEARNING_RATE);
    trainingTargets.add(tf.train.applyGradientDescent(weights, alpha, gradients.dy(0)));
    trainingTargets.add(tf.train.applyGradientDescent(biases, alpha, gradients.dy(1)));

    Operand<Long> predicted = tf.math.argMax(softmax, tf.constant(1));
    Operand<Long> expected = tf.math.argMax(labels, tf.constant(1));
    Operand<Float> accuracy = tf.math.mean(tf.dtypes.cast(tf.math.equal(predicted, expected), Float.class), constants(tf, 0));

    try (Session session = new Session(graph)) {

      // Initialize graph variables
      Session.Runner initialization = session.runner();
      initializationTargets.forEach(initialization::addTarget);
      initialization.run();

      // Train the graph
      for (Iterator<ImageBatch> batchIter = dataset.trainingBatchIterator(TRAINING_BATCH_SIZE); batchIter.hasNext();) {
        ImageBatch batch = batchIter.next();
        try (Tensor<Float> batchImages = Tensors.create(batch.images());
             Tensor<Float> batchLabels = Tensors.create(batch.labels())) {
          Session.Runner training = session.runner();
          trainingTargets.forEach(training::addTarget);
          training.feed(images.asOutput(), batchImages)
                  .feed(labels.asOutput(), batchLabels)
                  .run();
        }
      }

      // Test the graph
      ImageBatch testBatch = dataset.testBatch();
      try (Tensor<Float> testImages = Tensors.create(testBatch.images());
           Tensor<Float> testLabels = Tensors.create(testBatch.labels());
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
  private static final float LEARNING_RATE = 0.5f;
  
  private Graph graph;
  
  private SimpleMnist(Graph graph) {
    this.graph = graph;
  }

  // Helper that converts a single integer into an array
  private Operand<Integer> constants(Ops tf, int... i) {
    return tf.constant(i);
  }
}
