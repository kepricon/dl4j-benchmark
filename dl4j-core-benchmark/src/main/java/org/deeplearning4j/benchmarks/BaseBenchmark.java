package org.deeplearning4j.benchmarks;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.listeners.BenchmarkListener;
import org.deeplearning4j.listeners.BenchmarkReport;
import org.deeplearning4j.models.ModelSelector;
import org.deeplearning4j.models.ModelType;
import org.deeplearning4j.models.TestableModel;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.parallelism.ParallelWrapper;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.lang.reflect.Method;
import java.util.Map;

/**
 * Benchmarks popular CNN models using the CIFAR-10 dataset.
 */
@Slf4j
public abstract class BaseBenchmark {
    protected int listenerFreq = 10;
    protected int iterations = 1;
    protected static Map<ModelType,TestableModel> networks;
    protected boolean train = true;
    protected int maxIteration = 500;

    public void benchmarkCNN(int height, int width, int channels, int numLabels, int batchSize, int seed, String datasetName, DataSetIterator iter, ModelType modelType, int numGPUs) throws Exception {
        log.info("Building models for "+modelType+"....");
        networks = ModelSelector.select(modelType, height, width, channels, numLabels, seed, iterations);
        String dimensions = datasetName+" "+batchSize+"x"+channels+"x"+height+"x"+width;

        benchmark(dimensions, iter, numGPUs);
    }

    public void benchmarkRNN(String datasetName, DataSetIterator iter, ModelType modelType, int numGPUs) throws Exception {
        log.info("Building models for " + modelType + "....");
        networks = ModelSelector.select(modelType, 0, 0, 0, 0, 0, 0);
        String dimensions = datasetName;

        benchmark(dimensions, iter, numGPUs);
    }

    private void benchmark(String description, DataSetIterator iter, int numGPUs) throws Exception {
        long totalTime = System.currentTimeMillis();

        log.info("========================================");
        log.info("===== Benchmarking selected models =====");
        log.info("========================================");

        for (Map.Entry<ModelType, TestableModel> net : networks.entrySet()) {

            log.info("Selected: "+net.getKey().toString()+" "+description);
            Model model = net.getValue().init();
            BenchmarkReport report = new BenchmarkReport(net.getKey().toString(), description);
            report.setModel(model);

            model.setListeners(new ScoreIterationListener(listenerFreq), new BenchmarkListener(report));

            File locationToSave = new File(net.getKey().toString() +".zip");

            log.info("===== Benchmarking training iteration =====");
            long epochTime = System.currentTimeMillis();
            if (numGPUs == 0 || numGPUs == 1) { // cpu mode or single gpu mode
                int nIteration = 0;
                if (model instanceof MultiLayerNetwork) {
//                        ((MultiLayerNetwork) model).fit(iter);
                    while(iter.hasNext() && nIteration < maxIteration){
                        ((MultiLayerNetwork) model).fit(iter.next());
                        nIteration++;
                    }
                }else if (model instanceof ComputationGraph) {
//                        ((ComputationGraph) model).fit(iter);
                    while(iter.hasNext() && nIteration < maxIteration){
                        ((ComputationGraph) model).fit(iter.next());
                        nIteration++;
                    }
                }
            } else { // multiple gpu mode
                numGPUs = (numGPUs == -1) ? Nd4j.getAffinityManager().getNumberOfDevices() : numGPUs;
                ParallelWrapper pw = new ParallelWrapper.Builder<>(model)
                        .prefetchBuffer(numGPUs)
                        .reportScoreAfterAveraging(true)
                        .averagingFrequency(10)
                        .useLegacyAveraging(false)
                        .useMQ(true)
                        .workers(numGPUs)
                        .build();

                pw.fit(iter);
                pw.close();
            }

            epochTime = System.currentTimeMillis() - epochTime;
            report.setAvgEpochTime(epochTime);

            log.info("===== Benchmarking forward/backward pass =====");
            /*
                Notes: popular benchmarks will measure the time it takes to set the input and feed forward
                and backward. This is consistent with benchmarks seen in the wild like this code:
                https://github.com/jcjohnson/cnn-benchmarks/blob/master/cnn_benchmark.lua
             */
            iter.reset(); // prevents NPE
            long totalForward = 0;
            long totalBackward = 0;
            long nIterations = 0;
            if(model instanceof MultiLayerNetwork) {
                while(iter.hasNext() && nIterations < maxIteration) {
                    DataSet ds = iter.next();
                    INDArray input = ds.getFeatures();
                    INDArray labels = ds.getLabels();

                    // forward
                    long forwardTime = System.currentTimeMillis();
                    ((MultiLayerNetwork) model).setInput(input);
                    ((MultiLayerNetwork) model).setLabels(labels);
                    ((MultiLayerNetwork) model).feedForward();
                    forwardTime = System.currentTimeMillis() - forwardTime;
                    totalForward += forwardTime;

                    // backward
                    long backwardTime = System.currentTimeMillis();
                    Method m = MultiLayerNetwork.class.getDeclaredMethod("backprop"); // requires reflection
                    m.setAccessible(true);
                    m.invoke(model);
                    backwardTime = System.currentTimeMillis() - backwardTime;
                    totalBackward += backwardTime;

                    nIterations += 1;
                    if(nIterations % 100 == 0) log.info("Completed "+nIterations+" iterations");
                }
            }
            if(model instanceof ComputationGraph) {
                log.info(((ComputationGraph) model).getListeners().toString());
                log.info(((ComputationGraph) model).getListeners().size()+"");
                while(iter.hasNext() && nIterations < maxIteration) {
                    DataSet ds = iter.next();
                    INDArray input = ds.getFeatures();
                    INDArray labels = ds.getLabels();

                    // forward
                    long forwardTime = System.currentTimeMillis();
                    ((ComputationGraph) model).setInput(0, input);
                    ((ComputationGraph) model).setLabel(0, labels);
                    ((ComputationGraph) model).feedForward();
                    forwardTime = System.currentTimeMillis() - forwardTime;
                    totalForward += forwardTime;

                    // backward
                    long backwardTime = System.currentTimeMillis();
                    Method m = ComputationGraph.class.getDeclaredMethod("calcBackpropGradients", boolean.class, INDArray[].class);
                    m.setAccessible(true);
                    m.invoke(model, false, new INDArray[0]);
                    backwardTime = System.currentTimeMillis() - backwardTime;
                    totalBackward += backwardTime;

                    nIterations += 1;
                    if(nIterations % 100 == 0) log.info("Completed "+nIterations+" iterations");
                }
            }
            report.setAvgFeedForward((double) totalForward / (double) nIterations);
            report.setAvgBackprop((double) totalBackward / (double) nIterations);


            log.info("=============================");
            log.info("===== Benchmark Results =====");
            log.info("=============================");

            System.out.println(report.getModelSummary());
            System.out.println(report.toString());
        }
    }
}
