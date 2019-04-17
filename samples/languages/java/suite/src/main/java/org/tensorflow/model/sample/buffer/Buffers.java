package org.tensorflow.model.sample.buffer;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;

public class Buffers {

  public static ByteBuffer stringToBuffer(String value) throws IOException {
    return stringsToBuffer(Arrays.asList(value));
  }
   
  public static ByteBuffer stringsToBuffer(List<String> values) throws IOException {
    long offsets[] = new long[values.size()];
    byte[][] data = new byte[values.size()][];
    int dataSize = 0;

    // Convert strings to encoded bytes and calculate required data size, including a varint for each of them
    Iterator<String> valueIter = values.iterator();
    for (int i = 0; i < values.size(); ++i) {
      byte[] byteValue = valueIter.next().getBytes("UTF-8");
      data[i] = byteValue;
      int length = byteValue.length + varintLength(byteValue.length);
      dataSize += length;
      if (valueIter.hasNext()) {
        offsets[i + 1] = offsets[i] + length;
      }
    }

    // Important: buffer must follow native byte order
    ByteBuffer buffer = ByteBuffer.allocate(dataSize + (offsets.length * 8)).order(ByteOrder.nativeOrder());

    // Write offsets to the elements in the buffer
    for (int i = 0; i < offsets.length; ++i) {
      buffer.putLong(offsets[i]);
    }
    
    // Write data, where each elements is preceded by its length encoded as a varint
    for (int i = 0; i < data.length; ++i) {
      encodeVarint(buffer, data[i].length);
      buffer.put(data[i]);
    }

    return (ByteBuffer)buffer.rewind();
  }

  public static ByteBuffer stringsToBuffer(List<String> values, Character padding) throws IOException {
    int numElements = values.size();
    byte[][] data = new byte[numElements][];
    int elementSize = 0;

    // Convert strings to encoded bytes and calculate required data size, including a varint for each of them
    Iterator<String> valueIter = values.iterator();
    for (int i = 0; i < values.size(); ++i) {
      byte[] byteValue = valueIter.next().getBytes("UTF-8");
      data[i] = byteValue;
      if (byteValue.length > elementSize) {
        elementSize = byteValue.length;
      }
    }
    int varintLength = varintLength(elementSize);

    // Important: buffer must follow native byte order
    ByteBuffer buffer = ByteBuffer.allocate(numElements * (elementSize + 8 + varintLength)).order(ByteOrder.nativeOrder());

    // Write offsets to the elements in the buffer
    for (int i = 0; i < numElements; ++i) {
      buffer.putLong((elementSize + varintLength) * i);
    }
    
    // Write data, where each elements is preceded by its length encoded as a varint
    for (int i = 0; i < data.length; ++i) {
      encodeVarint(buffer, elementSize);
      buffer.put(data[i]);
      for (int j = 0; j < elementSize - data[i].length; ++j) {
        buffer.put((byte)padding.charValue());
      }
    }

    return (ByteBuffer)buffer.rewind();
  }
  
  private static void encodeVarint(ByteBuffer buffer, int value) {
      int v = value;
      while (v >= 0x80) {
        buffer.put((byte)((v & 0x7F) | 0x80));
        v >>= 7;
      }
      buffer.put((byte)v);
  }

  private static int varintLength(int length) {
    int len = 1;
    while (length >= 0x80) {
      length >>= 7;
      len++;
    }
    return len;
  }
}
