package org.tensorflow.model.sample.buffer;

import java.nio.ByteBuffer;
import java.nio.charset.Charset;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Shape;
import org.tensorflow.Tensor;
import org.tensorflow.Tensors;
import org.tensorflow.example.BytesList;
import org.tensorflow.example.Example;
import org.tensorflow.example.Feature;
import org.tensorflow.example.Features;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.ParseExample;
import org.tensorflow.op.core.Placeholder;

import com.google.protobuf.ByteString;

public class ParseExampleDemo {
  
  private static Example buildExample() throws Exception {
    Feature featureA = Feature.newBuilder()
        .setBytesList(BytesList.newBuilder()
            .addValue(ByteString.copyFrom("A feature", Charset.forName("UTF-8"))).build())
        .build();

    Feature featureB = Feature.newBuilder()
        .setBytesList(BytesList.newBuilder()
            .addValue(ByteString.copyFrom("Another feature", Charset.forName("UTF-8"))).build())
        .build();

    Features features = Features.newBuilder()
        .putFeature("featureA", featureA)
        .putFeature("featureB", featureB)
        .build();
    
    return Example.newBuilder().setFeatures(features).build();
  }

  public static void main(String[] args) throws Exception {
    Example example = buildExample(); 
    System.out.println("Example is: " + example.toString());
    ByteBuffer exampleData = Buffers.stringsToBuffer(Collections.singletonList(example.toByteString().toString("UTF-8")));

    try (Graph g = new Graph()) {
      Ops tf = Ops.create(g);

      Placeholder<String> serialized = tf.placeholder(String.class, Placeholder.shape(Shape.make(1)));
      Placeholder<String> names = tf.placeholder(String.class, Placeholder.shape(Shape.make(1)));
      
      ParseExample parser = tf.parseExample(
          serialized, 
          tf.constant(new byte[][] {"example".getBytes()}),
          Collections.emptyList(), 
          Arrays.asList(tf.constant("featureA"), tf.constant("featureB")),
          Arrays.asList(tf.constant(String.class, new long[]{1}, Buffers.stringToBuffer("oops")), 
                        tf.constant(String.class, new long[]{1}, Buffers.stringToBuffer("oopsies"))),
          Collections.emptyList(),
          Arrays.asList(Shape.make(1), Shape.make(1))
      );
      
      try (Session s = new Session(g)) {
        try (Tensor<String> exampleTensor = Tensor.create(String.class, new long[] {1L}, exampleData);
            Tensor<String> nameTensor = Tensors.create("")) {

          List<Tensor<?>> featureValues = s.runner()
            .feed(serialized.asOutput(), exampleTensor)
            .feed(names.asOutput(), nameTensor)
            .fetch(parser.denseValues().get(0))
            .fetch(parser.denseValues().get(1))
            .run();

          for (Tensor<?> featureValue : featureValues) {
            // Tricky way to read back a string from a tensor,
            // each feature is a single list (dim 0) of one string (dim 1) of a variable length (dim 2)
            byte[][][] value = new byte[1][1][];
            featureValue.copyTo(value);
            System.out.println(new String(value[0][0], "UTF-8"));
            featureValue.close();
          }
        }
      }

    } catch (Exception e) {
      System.err.println(e.toString());
      System.exit(-1);
    }
  }
}
