package util;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import org.tensorflow.Operand;
import org.tensorflow.Session;

public class ExecutionContext {
  
  public ExecutionContext addTarget(Operand<?> op) {
    targetsToDiscard.add(op.asOutput());
    return this;
  }
  
  public ExecutionContext addTargetToFetch(Operand<?> op) {
    targetsToFetch.add(op);
    return this;
  }
  
  public ExecutionStep createStep(Session session) {
    return new ExecutionStep(session, this);
  }  
  
  private final List<Operand<?>> targetsToDiscard = new ArrayList<>();
  private final List<Operand<?>> targetsToFetch = new ArrayList<>();
  
  List<Operand<?>> targetsToDiscard() {
    return Collections.unmodifiableList(targetsToDiscard);
  }

  List<Operand<?>> targetsToFetch() {
    return Collections.unmodifiableList(targetsToFetch);
  }
}
