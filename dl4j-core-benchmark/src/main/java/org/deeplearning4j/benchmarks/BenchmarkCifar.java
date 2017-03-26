package org.deeplearning4j.benchmarks;

import lombok.extern.slf4j.Slf4j;
import org.datavec.image.loader.CifarLoader;
import org.datavec.image.transform.FlipImageTransform;
import org.datavec.image.transform.ImageTransform;
import org.deeplearning4j.Utils.DL4J_Utils;
import org.deeplearning4j.datasets.iterator.MultipleEpochsIterator;
import org.deeplearning4j.datasets.iterator.impl.CifarDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.listeners.BenchmarkListener;
import org.deeplearning4j.listeners.BenchmarkReport;
import org.deeplearning4j.models.ModelSelector;
import org.deeplearning4j.models.ModelType;
import org.deeplearning4j.models.TestableModel;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.PerformanceListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.Option;
import org.nd4j.jita.conf.CudaEnvironment;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.IOException;
import java.text.DecimalFormat;
import java.util.Map;

/**
 * Benchmarks popular CNN models using the CIFAR-10 dataset.
 */
@Slf4j
public class BenchmarkCifar {

    // values to pass in from command line when compiled, esp running remotely
    @Option(name = "--modelType", usage = "Model type (e.g. ALEXNET, VGG16, or CNN).", aliases = "-mT")
    public String modelType = "INCEPTIONRESNETV1";
//    @Option(name="--numGPUs",usage="How many workers to use for multiple GPUs.",aliases = "-ng")
//    public int numGPUs = 0;
    @Option(name="--numTrainExamples",usage="Num train examples.",aliases = "-nTrain")
    public int numTrainExamples = CifarLoader.NUM_TRAIN_IMAGES;
    @Option(name="--numTestExamples",usage="Num test examples.",aliases = "-nTest")
    public int numTestExamples = CifarLoader.NUM_TEST_IMAGES;
    @Option(name="--trainBatchSize",usage="Train batch size.",aliases = "-nTrainB")
    public int trainBatchSize = 125;
    @Option(name="--testBatchSize",usage="Test batch size.",aliases = "-nTestB")
    public int testBatchSize = 125;
    @Option(name="--epochs",usage="Number of epochs.",aliases = "-epochs")
    public int epochs = 8;
    @Option(name="--preProcess",usage="Set preprocess.",aliases = "-pre")
    public boolean preProcess = true;
    @Option(name="--deviceCache",usage="Set CUDA device cache.",aliases = "-dcache")
    public long deviceCache = 6L;
    @Option(name="--hostCache",usage="Set CUDA host cache.",aliases = "-hcache")
    public long hostCache = 12L;
    @Option(name="--gcThreads",usage="Set Garbage Collection threads.",aliases = "-gcthreads")
    public int gcThreads = 5;
    @Option(name="--gcWindow",usage="Set Garbage Collection window in milliseconds.",aliases = "-gcwindow")
    public int gcWindow = 110;

    protected static int height = 160;
    protected static int width = 160;
    protected static int channels = 3;
    protected static int numLabels = CifarLoader.NUM_LABELS;

    protected static int seed = 42;
    protected static int listenerFreq = 10;
    protected static int iterations = 1
            ;
    protected static Map<ModelType,TestableModel> networks;
    protected boolean train = true;


    public void run(String[] args) throws IOException {
        long totalTime = System.currentTimeMillis();

        // Parse command line arguments if they exist
        CmdLineParser parser = new CmdLineParser(this);
        try {
            parser.parseArgument(args);
        } catch (CmdLineException e) {
            // handling of wrong arguments
            System.err.println(e.getMessage());
            parser.printUsage(System.err);
        }

        if(modelType == "ALL" || modelType == "RNN") {
            throw new UnsupportedOperationException("CIFAR-10 benchmarks are applicable to CNN models only.");
        }

        // memory management optimizations
        CudaEnvironment.getInstance().getConfiguration()
                .allowMultiGPU(false)
                .setMaximumDeviceCache(deviceCache * 1024L * 1024L * 1024L)
                .setMaximumHostCache(hostCache * 1024L * 1024L * 1024L)
                .setNumberOfGcThreads(gcThreads);
        Nd4j.create(1);
        Nd4j.getMemoryManager().togglePeriodicGc(true);
        Nd4j.getMemoryManager().setAutoGcWindow(gcWindow);
        Nd4j.getMemoryManager().setOccasionalGcFrequency(0);

        /*
        Primary benchmarking happens from here
         */
        log.info("Loading data...");
        ImageTransform flip = new FlipImageTransform(seed); // Should random flip some images but not all
        CifarDataSetIterator cifar = new CifarDataSetIterator(trainBatchSize, numTrainExamples, new int[]{height, width, channels}, numLabels, flip, preProcess, train);

        log.info("Building models for "+modelType+"....");
        networks = ModelSelector.select(ModelType.valueOf(modelType),height, width, channels, numLabels, seed, iterations);

        log.info("========================================");
        log.info("===== Benchmarking selected models =====");
        log.info("========================================");
        for (Map.Entry<ModelType, TestableModel> net : networks.entrySet()) {
            String dimensions = "CIFAR-10 "+trainBatchSize+"x"+channels+"x"+height+"x"+width;
            log.info("Selected: "+net.getKey().toString()+" "+dimensions);

            Model model = net.getValue().init();
            BenchmarkReport report = new BenchmarkReport(net.getKey().toString(), dimensions);
            report.setModel(model);

            model.setListeners(new ScoreIterationListener(listenerFreq), new BenchmarkListener(report));

            log.info("===== Benchmarking training iteration =====");
            if(model instanceof MultiLayerNetwork)
                ((MultiLayerNetwork) model).fit(cifar);
            if(model instanceof ComputationGraph)
                ((ComputationGraph) model).fit(cifar);

            log.info("===== Benchmarking forward pass =====");
            cifar.reset(); // prevents NPE
            long nIterations = 1000;
            INDArray input = cifar.next().getFeatures();
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

    public static void main(String[] args) throws Exception {
        new BenchmarkCifar().run(args);
    }
}
