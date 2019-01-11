package org.tensorflow.model.sample.mnist;

import java.util.ArrayList;
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
import org.tensorflow.op.core.Constant;
import org.tensorflow.op.core.Gradients;
import org.tensorflow.op.core.Placeholder;
import org.tensorflow.op.core.Variable;
import org.tensorflow.op.math.Mean;

public class MultiLayerMnist implements Runnable {

  public static void main(String[] args) {
    ImageDataset dataset = ImageDataset.create(VALIDATION_SIZE);
    try (Graph graph = new Graph()) {
      MultiLayerMnist mnist = new MultiLayerMnist(graph, dataset);
      mnist.run();
    }
  }

  @Override
  public void run() {
    Ops tf = Ops.create(graph);
    Context ctx = new Context();
    
    // Create placeholders
    Placeholder<Float> images = tf.placeholder(Float.class, Placeholder.shape(Shape.make(-1, 784)));
    Placeholder<Float> labels = tf.placeholder(Float.class);
    
    // Create hidden and output layers
    Operand<Float> hidden1 = tf.nn.relu(layer(tf, ctx, images, 784, 128));
    Operand<Float> hidden2 = tf.nn.relu(layer(tf, ctx, hidden1, 128, 32));
    Operand<Float> softmax = tf.nn.softmax(layer(tf, ctx, hidden2, 32, 10));

    // Compute loss and apply gradient backprop
    Mean<Float> crossEntropy = tf.math.mean(tf.math.neg(tf.reduceSum(tf.math.mul(labels, tf.math.log(softmax)), constArray(tf, 1))), constArray(tf, 0));
    Gradients gradients = tf.gradients(crossEntropy, ctx.variables);
    Constant<Float> alpha = tf.constant(LEARNING_RATE);
    for (int i = 0; i < ctx.variables.size(); ++i) {
      ctx.trainingTargets.add(tf.train.applyGradientDescent(ctx.variables.get(i), alpha, gradients.dy(i)));
    }

    Operand<Long> predicted = tf.math.argMax(softmax, tf.constant(1));
    Operand<Long> expected = tf.math.argMax(labels, tf.constant(1));
    Operand<Float> accuracy = tf.math.mean(tf.dtypes.cast(tf.math.equal(predicted, expected), Float.class), constArray(tf, 0));

    try (Session session = new Session(graph)) {

      // Initialize graph variables
      Session.Runner initialization = session.runner();
      ctx.variablesInit.forEach(initialization::addTarget);
      initialization.run();

      // Train the graph
      for (Iterator<ImageBatch> batchIter = dataset.trainingBatchIterator(TRAINING_BATCH_SIZE); batchIter.hasNext();) {
        ImageBatch batch = batchIter.next();
        try (Tensor<Float> batchImages = Tensors.create(batch.images());
             Tensor<Float> batchLabels = Tensors.create(batch.labels())) {
          Session.Runner training = session.runner();
          ctx.trainingTargets.forEach(training::addTarget);
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
  private static final float LEARNING_RATE = 0.2f;
  
  private Graph graph;
  private ImageDataset dataset;
    
  private MultiLayerMnist(Graph graph, ImageDataset dataset) {
    this.graph = graph;
    this.dataset = dataset;
  }
  
  private static class Context {
    List<Variable<Float>> variables = new ArrayList<>();
    List<Operand<Float>> variablesInit = new ArrayList<>();
    List<Operand<Float>> trainingTargets = new ArrayList<>();
  }
  
  private static Operand<Float> layer(Ops tf, Context ctx, Operand<Float> input, int inputSize, int layerSize) {
    Variable<Float> weights = tf.variable(Shape.make(inputSize, layerSize), Float.class);
    ctx.variables.add(weights);
    ctx.variablesInit.add(tf.assign(weights, tf.zeros(constArray(tf, inputSize, layerSize), Float.class)));

    Variable<Float> biases = tf.variable(Shape.make(layerSize), Float.class);
    ctx.variables.add(biases);
    ctx.variablesInit.add(tf.assign(biases, tf.zeros(constArray(tf, layerSize), Float.class)));
    
    return tf.math.add(tf.linalg.matMul(input, weights), biases);
  }

  // Helper that converts a single integer into an array
  private static Operand<Integer> constArray(Ops tf, int... i) {
    return tf.constant(i);
  }
}
