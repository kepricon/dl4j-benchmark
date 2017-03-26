package org.deeplearning4j.benchmarks;

import lombok.extern.slf4j.Slf4j;
import org.datavec.image.loader.CifarLoader;
import org.datavec.image.transform.FlipImageTransform;
import org.datavec.image.transform.ImageTransform;
import org.deeplearning4j.datasets.iterator.impl.CifarDataSetIterator;
import org.deeplearning4j.listeners.BenchmarkListener;
import org.deeplearning4j.listeners.BenchmarkReport;
import org.deeplearning4j.models.ModelSelector;
import org.deeplearning4j.models.ModelType;
import org.deeplearning4j.models.TestableModel;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.Option;
import org.nd4j.jita.conf.CudaEnvironment;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;

import java.io.IOException;
import java.util.Map;

/**
 * Benchmarks popular CNN models using the CIFAR-10 dataset.
 */
@Slf4j
public abstract class BaseBenchmark {
    protected int listenerFreq = 10;
    protected int iterations = 1
            ;
    protected static Map<ModelType,TestableModel> networks;
    protected boolean train = true;

    public void benchmark(int height, int width, int channels, int numLabels, int batchSize, int seed, String datasetName, DataSetIterator iter, ModelType modelType) {
        long totalTime = System.currentTimeMillis();

        log.info("Building models for "+modelType+"....");
        networks = ModelSelector.select(modelType,height, width, channels, numLabels, seed, iterations);

        log.info("========================================");
        log.info("===== Benchmarking selected models =====");
        log.info("========================================");

        for (Map.Entry<ModelType, TestableModel> net : networks.entrySet()) {
            String dimensions = datasetName+" "+batchSize+"x"+channels+"x"+height+"x"+width;
            log.info("Selected: "+net.getKey().toString()+" "+dimensions);

            Model model = net.getValue().init();
            BenchmarkReport report = new BenchmarkReport(net.getKey().toString(), dimensions);
            report.setModel(model);

            model.setListeners(new ScoreIterationListener(listenerFreq), new BenchmarkListener(report));

            log.info("===== Benchmarking training iteration =====");
            if(model instanceof MultiLayerNetwork)
                ((MultiLayerNetwork) model).fit(iter);
            if(model instanceof ComputationGraph)
                ((ComputationGraph) model).fit(iter);

            log.info("===== Benchmarking forward pass =====");
            iter.reset(); // prevents NPE
            long nIterations = 1000;
            INDArray input = iter.next().getFeatures();
            if(model instanceof MultiLayerNetwork) {
                ((MultiLayerNetwork) model).setInput(input);
            }
            if(model instanceof ComputationGraph) {
                ((ComputationGraph) model).setInput(0, input);
            }

            long forwardTime = System.currentTimeMillis();
            for (int i = 0; i < nIterations; i++) {
                if(model instanceof MultiLayerNetwork)
                    ((MultiLayerNetwork) model).feedForward();
                if(model instanceof ComputationGraph)
                    ((ComputationGraph) model).feedForward();
            }
            forwardTime = System.currentTimeMillis() - forwardTime;
            report.setAvgFeedForward(forwardTime / nIterations);

            log.info("=============================");
            log.info("===== Benchmark Results =====");
            log.info("=============================");

            System.out.println(report.getModelSummary());
            System.out.println(report.toString());
        }
    }
}
