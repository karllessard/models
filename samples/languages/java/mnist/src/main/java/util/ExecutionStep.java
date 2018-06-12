package util;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import org.tensorflow.Operand;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.op.core.Placeholder;

public class ExecutionStep implements AutoCloseable {

  public <T> ExecutionStep feed(Placeholder<T> placeholder, Tensor<T> value) {
    return feed(placeholder, value, true);
  }
  
  public <T> ExecutionStep feed(Placeholder<T> placeholder, Tensor<T> value, boolean owner) {
    sessionRunner.feed(placeholder.asOutput(), value);
    if (owner) {
      ownedResources.add(value);
    }
    return this;
  }
  
  public Tensor<?> target(String opName) {
    for (int idx = 0; idx < this.context.targetsToFetch().size(); ++idx) {
      if (context.targetsToFetch().get(idx).asOutput().op().name().equals(opName)) {
        return fetchedTargets.get(idx);
      }
    }
    throw new IllegalArgumentException("No target \"" + opName + "\" has been fetched");
  }

  public Tensor<?> target(Operand<?> operand) {
    return target(operand.asOutput().op().name());
  }
  
  public void run() {
    context.targetsToDiscard().forEach(sessionRunner::addTarget);
    context.targetsToFetch().forEach(sessionRunner::fetch);
    fetchedTargets = sessionRunner.run();
  }

  @Override
  public void close() {
    if (fetchedTargets != null) {
      fetchedTargets.forEach(ExecutionStep::closeResource);
    }
    ownedResources.forEach(ExecutionStep::closeResource);
  }
   
  private final ExecutionContext context;
  private Session.Runner sessionRunner;
  private final List<Tensor<?>> ownedResources = new ArrayList<>();
  private List<Tensor<?>> fetchedTargets = Collections.emptyList();
  
  ExecutionStep(Session session, ExecutionContext context) {
    sessionRunner = session.runner();
    this.context = context;
  }

  private static void closeResource(Tensor<?> tensor) {
    try {
      tensor.close();
    } catch (Exception e) {
      throw new RuntimeException(e);
    }
  }
}