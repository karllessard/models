package org.tensorflow.model.sample.mnist;

import java.util.Collections;

import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Constant;
import org.tensorflow.op.summary.CloseSummaryWriter;
import org.tensorflow.op.summary.CreateSummaryFileWriter;
import org.tensorflow.op.summary.MergeSummary;
import org.tensorflow.op.summary.ScalarSummary;
import org.tensorflow.op.summary.SummaryWriter;
import org.tensorflow.op.summary.WriteScalarSummary;

public class SummaryTest implements Runnable {

  public static void main(String[] args) {
    try (Graph graph = new Graph()) {
      SummaryTest summaryTest = new SummaryTest(graph);
      summaryTest.run();
    }
  }

  @Override
  public void run() {
    SummaryWriter summaryWriter = SummaryWriter.create(ops.scope());
    Constant<Float> scalar = ops.constant(1.5f);
    Constant<String> tag = ops.constant("/test/myvalue");
    ScalarSummary summary = ops.summary.scalarSummary(tag, scalar);
    MergeSummary mergeSummary = ops.summary.mergeSummary(Collections.singletonList(summary));

    try (Session session = new Session(graph)) {
      CreateSummaryFileWriter.create(ops.scope(),
          summaryWriter,
          str("/tmp/tensorflow/test/summary"),
          ops.constant(0), 
          ops.constant(0),
          str(".test")
      );
      session.runner().addTarget("CreateSummaryFileWriter").run(); // FIXME should be able to retrieve operation out of PrimitiveOp instead of relying on name?
      
      WriteScalarSummary.create(ops.scope(),
          summaryWriter, 
          ops.constant(1L), 
          tag,
          scalar
      );
      session.runner().addTarget(scalar).addTarget("WriteScalarSummary").run();

      CloseSummaryWriter.create(ops.scope(), summaryWriter);
      session.runner().addTarget("CloseSummaryWriter").run();
/*
      Session.Runner initialization = session.runner();
      // FIXME why are those ops hidden?
      SummaryWriter summaryWriter = SummaryWriter.create(ops.scope());
      CreateSummaryFileWriter.create(ops.scope().withName("summaryFileWriterCreate"), 
          summaryWriter, 
          str("/tmp/tensorflow/test/summary"),
          ops.constant(0), 
          ops.constant(0),
          str(".test")
      ); // FIXME should be able to retrieve operation out of PrimitiveOp instead of relying on name?
      initialization.addTarget("summaryFileWriterCreate");
      initialization.run();

      Session.Runner step = session.runner();
      WriteSummary.create(ops.scope().withName("writeSummary"), 
          summaryWriter, 
          ops.constant(1L), 
          scalar, 
          tag,
          mergeSummary
      );
      step.addTarget(scalar);
      step.addTarget("writeSummary");
      step.run();
      
      Session.Runner cleanup = session.runner();
      CloseSummaryWriter.create(ops.scope().withName("summaryFileWriterClose"), summaryWriter);
      cleanup.addTarget("summaryFileWriterClose");
      cleanup.run();
 */     
      try {
        Thread.sleep(1000);
      } catch (InterruptedException e) {
        // TODO Auto-generated catch block
        e.printStackTrace();
      }
    }
  }
  
  private Graph graph;
  private Ops ops;
    
  private SummaryTest(Graph graph) {
    this.graph = graph;
    this.ops = Ops.create(graph);
  }
  
  private Constant<String> str(String name) {
    return ops.constant(name); // TODO add name() ops in TF core ops, appending a label to current scope name prefix
  }
}
