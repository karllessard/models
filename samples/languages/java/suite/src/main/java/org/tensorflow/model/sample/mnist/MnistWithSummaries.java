package org.tensorflow.model.sample.mnist;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.function.BiFunction;
import java.util.stream.IntStream;

import org.tensorflow.Graph;
import org.tensorflow.Operand;
import org.tensorflow.Operation;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.Tensors;
import org.tensorflow.model.sample.mnist.data.ImageBatch;
import org.tensorflow.model.sample.mnist.data.ImageDataset;
import org.tensorflow.op.NnOps;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Constant;
import org.tensorflow.op.core.Gradients;
import org.tensorflow.op.core.Placeholder;
import org.tensorflow.op.core.Reshape;
import org.tensorflow.op.core.Variable;
import org.tensorflow.op.math.Mean;
import org.tensorflow.op.summary.CloseSummaryWriter;
import org.tensorflow.op.summary.CreateSummaryFileWriter;
import org.tensorflow.op.summary.SummaryWriter;
import org.tensorflow.op.summary.WriteHistogramSummary;
import org.tensorflow.op.summary.WriteImageSummary;
import org.tensorflow.op.summary.WriteScalarSummary;
import org.tensorflow.types.TFloat32;
import org.tensorflow.types.TInt32;
import org.tensorflow.types.TInt64;
import org.tensorflow.types.TString;
import org.tensorflow.types.TUInt8;
import org.tensorflow.tools.Shape;

public class MnistWithSummaries implements Runnable {

  public static void main(String[] args) {
    ImageDataset dataset = ImageDataset.create(VALIDATION_SIZE);
    try (Graph graph = new Graph()) {
      MnistWithSummaries mnist = new MnistWithSummaries(graph, dataset);
      long start = System.currentTimeMillis();
      mnist.run();
      long end = System.currentTimeMillis();
      System.out.println("Took " + (((float)(end - start))/ 1000.0f) + " sec to execute");
    }
  }

  @Override
  public void run() {
    Ops tf = Ops.create(graph);
    Context ctx = new Context();
    ctx.summaryWriter = SummaryWriter.create(tf.scope());
    
    // Create placeholders
    Ops tfIn = tf.withSubScope("input");
    Placeholder<TFloat32> images = tfIn.placeholder(
        TFloat32.DTYPE, Placeholder.shape(Shape.make(-1, 784)));
    Placeholder<TFloat32> labels = tfIn.placeholder(TFloat32.DTYPE);
    Placeholder<TInt64> step = tfIn.placeholder(TInt64.DTYPE, Placeholder.shape(Shape.scalar()));

    Reshape<TFloat32> thumbnails = tfIn.reshape(images, constArray(tfIn, -1, 28, 28, 1));
    ctx.summaries.add(WriteImageSummary.create(tfIn.scope(),
        ctx.summaryWriter, 
        step, 
        tag(tfIn, "thumbnails"), 
        thumbnails, 
        tfIn.constant(new byte[] {0, 0, 0}, TUInt8.DTYPE),
        WriteImageSummary.maxImages(10L)
    ).op());
    
    // Create layers and dropouts
    Operand<TFloat32> hidden = layer(tf.withSubScope("hidden"), ctx, step, images, 784, 500, NnOps::relu);
    Operand<TFloat32> output = layer(tf.withSubScope("output"), ctx, step, hidden, 500, 10, NnOps::softmax);

    // Compute loss and apply gradient backpropagation
    Operand<TFloat32> crossEntropy = crossEntropy(tf.withSubScope("cross_entropy"), labels, output);
    optimize(tf.withSubScope("train"), ctx, crossEntropy);
    
    // Compute accuracy of the predications for this batch
    Operand<TFloat32> accuracy = computeAccuracy(tf.withSubScope("accuracy"), ctx, step, labels, output);
    
    try (Session session = new Session(graph)) {

      // Initialize graph variables
      Session.Runner initialization = session.runner();
      initialization.addTarget(CreateSummaryFileWriter.create(tf.scope(), 
          ctx.summaryWriter, 
          tf.constant("/tmp/tensorflow/java/mnist"),
          tf.constant(1),
          tf.constant(1),
          tf.constant(".train")
      ).op());
      ctx.initTargets.forEach(initialization::addTarget);
      initialization.run();

      long stepNo = 1L;
      Iterator<ImageBatch> trainBatches = dataset.trainingBatchIterator(TRAINING_BATCH_SIZE);

      while (trainBatches.hasNext()) {
        Session.Runner training = session.runner()
            .feed(step.asOutput(), Tensors.create(stepNo++));

        if (stepNo % 10 == 0) {
          // Test graph
          ImageBatch testBatch = dataset.testBatch();
          try (Tensor<TFloat32> batchImages = Tensor.create(testBatch.shape(784), testBatch.images());
               Tensor<TFloat32> batchLabels = Tensor.create(testBatch.shape(10), testBatch.labels());
               Tensor<?> accuracyValue = training
                   .fetch(accuracy)
                   .feed(images, batchImages)
                   .feed(labels, batchLabels)
                   .run()
                   .get(0)) {
            System.out.println("Accuracy at step " + stepNo + ": " + accuracyValue.floatValue());
          }
        } else {
          // Train graph
          ctx.trainingTargets.forEach(training::addTarget);
          ctx.summaries.forEach(training::addTarget);
          ImageBatch batch = trainBatches.next();
          try (Tensor<TFloat32> batchImages = Tensor.create(batch.shape(784), batch.images());
               Tensor<TFloat32> batchLabels = Tensor.create(batch.shape(10), batch.labels())) {
             training
                 .feed(images, batchImages)
                 .feed(labels, batchLabels)
                 .run();
          }
        }
      }
      
      // Cleanup training
      Session.Runner cleanup = session.runner();
      cleanup.addTarget(CloseSummaryWriter.create(tf.scope(), ctx.summaryWriter).op());
      cleanup.run();
    }
  }
  
  private static final int VALIDATION_SIZE = 0;
  private static final int TRAINING_BATCH_SIZE = 100;
  private static final float LEARNING_RATE = 0.2f;
  
  private class Context {
    SummaryWriter summaryWriter;
    List<Variable<TFloat32>> variables = new ArrayList<>();
    List<Operand<?>> initTargets = new ArrayList<>();
    List<Operand<?>> trainingTargets = new ArrayList<>();
    List<Operation> summaries = new ArrayList<>();
  }
  
  private Graph graph;
  private ImageDataset dataset;
    
  private MnistWithSummaries(Graph graph, ImageDataset dataset) {
    this.graph = graph;
    this.dataset = dataset;
  }
  
  private Operand<TFloat32> layer(Ops tf,
      Context ctx,
      Operand<TInt64> step,
      Operand<TFloat32> input,
      int inputSize, 
      int layerSize, 
      BiFunction<NnOps, Operand<TFloat32>, Operand<TFloat32>> activation) {

    Ops tfW = tf.withSubScope("weights");
    Variable<TFloat32> weights = tfW.variable(Shape.make(inputSize, layerSize), TFloat32.DTYPE);
    ctx.variables.add(weights);
    ctx.initTargets.add(tfW.assign(weights, tfW.random.parameterizedTruncatedNormal(
        constArray(tfW, inputSize, layerSize), 
        tfW.constant(0.0f), 
        tfW.constant(0.1f),
        tfW.constant(-0.2f),
        tfW.constant(0.2f)
    )));
    summarize(tfW, ctx, step, weights);

    Ops tfB = tf.withSubScope("biases");
    Variable<TFloat32> biases = tfB.variable(Shape.make(layerSize), TFloat32.DTYPE);
    ctx.variables.add(biases);
    ctx.initTargets.add(tfB.assign(biases, tfB.fill(constArray(tfB, layerSize), tfB.constant(0.1f))));
    summarize(tfB, ctx, step, biases);
    
    Ops tfM = tf.withSubScope("Wx_plus_b");
    Operand<TFloat32> outputs = tfM.math.add(tfM.linalg.matMul(input, weights), biases);
    ctx.summaries.add(WriteHistogramSummary.create(tfM.scope(),
        ctx.summaryWriter,
        step,
        tag(tfM, "outputs"),
        outputs
    ).op());

    if (activation != null) {
      outputs = activation.apply(tf.nn, outputs);
      ctx.summaries.add(WriteHistogramSummary.create(tfM.scope(),
          ctx.summaryWriter,
          step,
          tag(tfM, "activated_outputs"),
          outputs
      ).op());
    }
    return outputs;
  }
  
  private void summarize(Ops tf, Context ctx, Operand<TInt64> step, Operand<TFloat32> var) {
    Constant<TInt32> axes = axesOf(tf, var);
    Mean<TFloat32> mean = tf.math.mean(var, axes);
    ctx.summaries.add(WriteScalarSummary.create(tf.scope(),
        ctx.summaryWriter, 
        step, 
        tag(tf, "mean"), 
        mean
    ).op());
    ctx.summaries.add(WriteScalarSummary.create(tf.scope(),
        ctx.summaryWriter, 
        step, 
        tag(tf, "stddev"), 
        tf.math.sqrt(tf.math.mean(tf.math.square(tf.math.sub(var, mean)), axes))
    ).op());
    ctx.summaries.add(WriteScalarSummary.create(tf.scope(),
        ctx.summaryWriter, 
        step, 
        tag(tf, "max"), 
        tf.reduceMax(var, axes)
    ).op());
    ctx.summaries.add(WriteScalarSummary.create(tf.scope(),
        ctx.summaryWriter, 
        step, 
        tag(tf, "min"), 
        tf.reduceMin(var, axes)
    ).op());
    ctx.summaries.add(WriteHistogramSummary.create(tf.scope(),
        ctx.summaryWriter,
        step,
        tag(tf, "histogram"),
        var
    ).op());
  }
  
  private Operand<TFloat32> crossEntropy(Ops tf, Operand<TFloat32> labels, Operand<TFloat32> logits) {
      return tf.math.mean(tf.math.neg(tf.reduceSum(tf.math.mul(labels, tf.math.log(logits)), constArray(tf, 1))), constArray(tf, 0));
  }
  
  private void optimize(Ops tf, Context ctx, Operand<TFloat32> loss) {
    Gradients gradients = tf.gradients(loss, ctx.variables);
    Constant<TFloat32> alpha = tf.constant(LEARNING_RATE);
    for (int i = 0; i < ctx.variables.size(); ++i) {
      ctx.trainingTargets.add(tf.train.applyGradientDescent(ctx.variables.get(i), alpha, gradients.dy(i)));
    }
  }
  
  private Operand<TFloat32> computeAccuracy(Ops tf, Context ctx, Operand<TInt64> step, Operand<TFloat32> labels, Operand<TFloat32> logits) {
    Operand<TInt64> predicted = tf.math.argMax(logits, tf.constant(1));
    Operand<TInt64> expected = tf.math.argMax(labels, tf.constant(1));
    Operand<TFloat32> accuracy = tf.math.mean(tf.dtypes.cast(tf.math.equal(predicted, expected), TFloat32.DTYPE), constArray(tf, 0));
    ctx.trainingTargets.add(accuracy);
    ctx.summaries.add(WriteScalarSummary.create(tf.scope(),
        ctx.summaryWriter, 
        step, 
        tag(tf, "accuracy"), 
        accuracy
    ).op());
    return accuracy;
  }
  
  // Helper to return axes of a tensor as an array
  private static Constant<TInt32> axesOf(Ops tf, Operand<?> tensor) {
    return tf.constant(IntStream.range(0, tensor.asOutput().shape().numDimensions()).toArray());
  }
  
  // Helper to create a name constant under current scope prefix
  private static Constant<TString> tag(Ops tf, String name) {
    return tf.constant(tf.scope().makeOpName(name));
  }

  // Helper that converts a single integer into an array
  private Operand<TInt32> constArray(Ops tf, int... i) {
    return tf.constant(i);
  }
}
