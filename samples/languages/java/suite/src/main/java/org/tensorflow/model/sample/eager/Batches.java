package org.tensorflow.data;

import java.util.ArrayList;
import java.util.Collections;
import org.tensorflow.EagerSession;
import org.tensorflow.Operand;
import org.tensorflow.Tensor;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Batch;
import org.tensorflow.op.core.Constant;

import java.nio.FloatBuffer;
import java.util.Arrays;
import java.util.List;

public class Batches {

  public static void testBatchWithBatchSize(long batchSize, float[] tensorData) {
    try (EagerSession env = EagerSession.create()) {
      Ops tf = Ops.create(env);

      List<Operand<?>> tensors = new ArrayList<>();
      tensors.add(Constant.create(tf.scope(), tensorData));
      tensors.add(Constant.create(tf.scope(), tensorData));

      // Test tf.batch. Leaving numBatchThreads, batchTimeoutMicros, gradTimeoutMicros = 1
      Batch batch = tf.batch(tensors, 1L, batchSize, 1L, 1L);

      // Retrieve batched tensor
      Tensor<?> batchedTensor = batch.batchedTensors().get(0).tensor();
      FloatBuffer buffer = FloatBuffer.allocate(tensorData.length * 2);
      batchedTensor.writeTo(buffer);

      System.out.println("Batched Tensor Count: " + batch.batchedTensors().size());
      System.out.println("Batched Tensor Shape: " + Arrays.toString(batchedTensor.shape().asArray()));
      System.out.println("Batched Tensor Contents: " + Arrays.toString(buffer.array()));
    }
  }

  public static void main(String[] args) {

    float[] data = new float[] {5, 6, 2, 4, 8, 4, 3, 7};

    // Batch size 8 (works)
    System.out.println("Batch Size: 8");
    testBatchWithBatchSize(8, data);

    System.out.println();

    // Batch size 4 (fails)
    System.out.println("Batch Size: 4");
    testBatchWithBatchSize(4, data);
  }
}