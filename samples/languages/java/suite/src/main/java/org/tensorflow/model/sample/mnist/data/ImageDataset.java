package org.tensorflow.model.sample.mnist.data;

import java.io.DataInputStream;
import java.io.IOException;
import java.nio.FloatBuffer;
import java.util.Arrays;
import java.util.Iterator;
import java.util.zip.GZIPInputStream;

public class ImageDataset {
  
  public class ImageBatchIterator implements Iterator<ImageBatch> {

    @Override
    public boolean hasNext() {
      return batchStart < totalSize();
    }

    @Override
    public ImageBatch next() {
      int size = Math.min(batchSize, images.length - batchStart);
      ImageBatch batch = new ImageBatch(
          serializeToBuffer(images, batchStart, size),
          serializeToBuffer(labels, batchStart, size),
          size
      );
      batchStart += batchSize;
      return batch;
    }
    
    ImageBatchIterator(int batchSize, float[][] images, float[][] labels) {
      this.batchStart = 0;
      this.batchSize = batchSize;
      this.images = images;
      this.labels = labels;
    }
    
    private int batchStart;
    private int batchSize;
    private float[][] images;
    private float[][] labels;
    
    private int totalSize() {
      return images.length;
    }
  }
  
  public static ImageDataset create(int validationSize) {
    try {
      float[][] trainImages = extractImages(TRAIN_IMAGES_ARCHIVE);
      float[][] trainLabels = extractLabels(TRAIN_LABELS_ARCHIVE, NUM_CLASSES);
      float[][] testImages = extractImages(TEST_IMAGES_ARCHIVE);
      float[][] testLabels = extractLabels(TEST_LABELS_ARCHIVE, NUM_CLASSES);
      
      if (validationSize > 0) {
        return new ImageDataset(
            Arrays.copyOfRange(trainImages, validationSize, trainImages.length),
            Arrays.copyOfRange(trainLabels, validationSize, trainLabels.length),
            Arrays.copyOfRange(trainImages, 0, validationSize),
            Arrays.copyOfRange(trainLabels, 0, validationSize),
            testImages,
            testLabels
        );
      }
      return new ImageDataset(trainImages, trainLabels, new float[][] {}, new float[][] {}, testImages, testLabels); 
      
    } catch (IOException e) {
      throw new AssertionError(e);
    }
  }
  
  public Iterator<ImageBatch> trainingBatchIterator(int batchSize) {
    return new ImageBatchIterator(batchSize, trainingImages, trainingLabels);
  }

  public Iterator<ImageBatch> validationBatchIterator(int batchSize) {
    return new ImageBatchIterator(batchSize, validationImages, validationLabels);
  }
  
  public ImageBatch testBatch() {
    return new ImageBatch(
        serializeToBuffer(testImages, 0, testImages.length), 
        serializeToBuffer(testLabels, 0, testLabels.length),
        testImages.length
    );
  }
  
  private static final String TRAIN_IMAGES_ARCHIVE = "train-images-idx3-ubyte.gz";
  private static final String TRAIN_LABELS_ARCHIVE = "train-labels-idx1-ubyte.gz";
  private static final String TEST_IMAGES_ARCHIVE = "t10k-images-idx3-ubyte.gz";
  private static final String TEST_LABELS_ARCHIVE = "t10k-labels-idx1-ubyte.gz";
  private static final Integer NUM_CLASSES = 10;
  private static final Integer IMAGE_ARCHIVE_MAGIC = 2051;
  private static final Integer LABEL_ARCHIVE_MAGIC = 2049;

  private final float[][] trainingImages;
  private final float[][] trainingLabels;
  private final float[][] validationImages;
  private final float[][] validationLabels;
  private final float[][] testImages;
  private final float[][] testLabels;
  
  private ImageDataset(float[][] trainingImages, float[][] trainingLabels, float[][] validationImages,
      float[][] validationLabels, float[][] testImages, float[][] testLabels) {
    this.trainingImages = trainingImages;
    this.trainingLabels = trainingLabels;
    this.validationImages = validationImages;
    this.validationLabels = validationLabels;
    this.testImages = testImages;
    this.testLabels = testLabels;
  }
  
  private static float[][] extractImages(String archiveName) throws IOException {
    DataInputStream archiveStream = 
        new DataInputStream(new GZIPInputStream(ImageDataset.class.getClassLoader().getResourceAsStream(archiveName)));
    int magic = archiveStream.readInt();
    if (!IMAGE_ARCHIVE_MAGIC.equals(magic)) {
      throw new IllegalArgumentException("\"" + archiveName + "\" is not a valid image archive");
    }
    int imageCount = archiveStream.readInt();
    int imageRows = archiveStream.readInt();
    int imageCols = archiveStream.readInt();
    System.out.println(String.format("Extracting %d images of %dx%d from %s", imageCount, imageRows, imageCols, archiveName));
    float[][] images = new float[imageCount][imageRows * imageCols];
    byte[] imageBuffer = new byte[imageRows * imageCols];
    for (int i = 0; i < imageCount; ++i) {
      archiveStream.readFully(imageBuffer);
      images[i] = toNormalizedVector(imageBuffer);
    }
    return images;
  }

  private static float[][] extractLabels(String archiveName, int numClasses) throws IOException {
    DataInputStream archiveStream = 
        new DataInputStream(new GZIPInputStream(ImageDataset.class.getClassLoader().getResourceAsStream(archiveName)));
    int magic = archiveStream.readInt();
    if (!LABEL_ARCHIVE_MAGIC.equals(magic)) {
      throw new IllegalArgumentException("\"" + archiveName + "\" is not a valid image archive");
    }
    int labelCount = archiveStream.readInt();
    System.out.println(String.format("Extracting %d labels from %s", labelCount, archiveName));
    byte[] labelBuffer = new byte[labelCount];
    archiveStream.readFully(labelBuffer);
    float[][] floats = new float[labelCount][10];
    for (int i = 0; i < labelCount; ++i) {
      floats[i] = toOneHotVector(10, labelBuffer[i]);
    }
    return floats;
  }

  private static float[] toOneHotVector(int numClasses, byte label) {
    FloatBuffer buf = FloatBuffer.allocate(numClasses);
    buf.put((int)(label & 0xFF), 1.0f);
    return buf.array();
  }
  
  private static float[] toNormalizedVector(byte[] bytes) {
    float[] floats = new float[bytes.length];
    for (int i = 0; i < bytes.length; ++i) {
      floats[i] = ((float)(bytes[i] & 0xFF)) / 255.0f;
    }
    return floats;
  }
  
  private static FloatBuffer serializeToBuffer(float[][] src, int start, int length) {
    FloatBuffer buffer = FloatBuffer.allocate(length * src[0].length);
    for (int i = start; i < start + length; ++i) {
      buffer.put(src[i]);
    }
    return (FloatBuffer) buffer.rewind();
  }
}
