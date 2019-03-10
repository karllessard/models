package org.tensorflow.model.sample.mnist.data;

import java.nio.FloatBuffer;

public class ImageBatch {
  
  public FloatBuffer images() {
    return images;
  }
  
  public FloatBuffer labels() {
    return labels;
  }
  
  public long[] shape(int elementSize) {
    return new long[] { numElements, elementSize };
  }
  
  public int size() {
    return numElements;
  }

  private final FloatBuffer images;
  private final FloatBuffer labels;
  private final int numElements;
  
  ImageBatch(FloatBuffer images, FloatBuffer labels, int numElements) {
    this.images = images;
    this.labels = labels;
    this.numElements = numElements;
  }
}
