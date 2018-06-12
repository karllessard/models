package mnist.data;

public class ImageBatch {
  
  public float[][] images() {
    return images;
  }
  
  public float[][] labels() {
    return labels;
  }
  
  public int size() {
    return labels.length;
  }

  private final float[][] images;
  private final float[][] labels;
  
  ImageBatch(float[][] images, float[][] labels) {
    this.images = images;
    this.labels = labels;
  }
}
