package org.deeplearning4j.benchmarks;

import lombok.extern.slf4j.Slf4j;
import org.datavec.image.loader.CifarLoader;
import org.datavec.image.transform.FlipImageTransform;
import org.datavec.image.transform.ImageTransform;
import org.deeplearning4j.Utils.DL4J_Utils;
import org.deeplearning4j.datasets.iterator.MultipleEpochsIterator;
import org.deeplearning4j.datasets.iterator.impl.CifarDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.models.ModelSelector;
import org.deeplearning4j.models.ModelType;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.Option;
import org.nd4j.jita.conf.CudaEnvironment;
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
    @Option(name = "--modelType", usage = "Model type (e.g. ALEXNET, VGG16, or ALL).", aliases = "-mT")
    public String modelType = "CAFFE_FULL_SIGMOID";
//    @Option(name="--numGPUs",usage="How many workers to use for multiple GPUs.",aliases = "-ng")
//    public int numGPUs = 0;
    @Option(name="--numTrainExamples",usage="Num train examples.",aliases = "-nTrain")
    public int numTrainExamples = CifarLoader.NUM_TRAIN_IMAGES;
    @Option(name="--numTestExamples",usage="Num test examples.",aliases = "-nTest")
    public int numTestExamples = CifarLoader.NUM_TEST_IMAGES;
    @Option(name="--trainBatchSize",usage="Train batch size.",aliases = "-nTrainB")
    public int trainBatchSize = 100;
    @Option(name="--testBatchSize",usage="Test batch size.",aliases = "-nTestB")
    public int testBatchSize = 100;
    @Option(name="--epochs",usage="Number of epochs.",aliases = "-epochs")
    public int epochs = 8;
    @Option(name="--preProcess",usage="Set preprocess.",aliases = "-pre")
    public boolean preProcess = true;
    @Option(name="--deviceCache",usage="Set CUDA device cache.",aliases = "-dcache")
    public long deviceCache = 6L;
    @Option(name="--hostCache",usage="Set CUDA host cache.",aliases = "-hcache")
    public long hostCache = 12L;
    @Option(name="--gcThreads",usage="Set Garbage Collection threads.",aliases = "-gcthreads")
    public long gcThreads = 5;
    @Option(name="--gcWindow",usage="Set Garbage Collection window in milliseconds.",aliases = "-gcwindow")
    public long gcWindow = 110;

    protected static int height = 32;
    protected static int width = 32;
    protected static int channels = 3;
    protected static int numLabels = CifarLoader.NUM_LABELS;

    protected static int seed = 42;
    protected static int listenerFreq = 1;
    protected static int iterations = 1
            ;
    protected static Map<ModelType,MultiLayerNetwork> networks;
    protected boolean train = true;

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

        log.debug("Load data...");
        ImageTransform flip = new FlipImageTransform(seed); // Should random flip some images but not all
        CifarDataSetIterator cifar = new CifarDataSetIterator(trainBatchSize, numTrainExamples, new int[]{height, width, channels}, numLabels, flip, preProcess, train);
        MultipleEpochsIterator iter = new MultipleEpochsIterator(epochs, cifar);

        log.debug("Build model....");
        networks = ModelSelector.select(ModelType.valueOf(modelType),height, width, channels, numLabels, seed, iterations);
        for (Map.Entry<ModelType, MultiLayerNetwork> net : networks.entrySet()) net.getValue().setListeners(new ScoreIterationListener(listenerFreq));

        log.debug("Train models...");
        long trainTime = System.currentTimeMillis();
        for (Map.Entry<ModelType, MultiLayerNetwork> net : networks.entrySet()) {
            net.getValue().fit(iter);
        }
        trainTime = System.currentTimeMillis() - trainTime;

        log.info("Evaluate models....");
        long testTime = System.currentTimeMillis();
        cifar.test(numTestExamples, testBatchSize);
        epochs = 1;
        iter = new MultipleEpochsIterator(epochs, cifar);
        for (Map.Entry<ModelType, MultiLayerNetwork> net : networks.entrySet()) {
            Evaluation eval = net.getValue().evaluate(iter);
            DecimalFormat df = new DecimalFormat("#.####");
            log.debug(eval.stats(true));
            log.info(df.format(eval.accuracy()));
        }
        testTime =  System.currentTimeMillis() - testTime;
        totalTime = System.currentTimeMillis() - totalTime;

        log.info("****************Example finished********************");
        DL4J_Utils.printTime("Train", trainTime);
        DL4J_Utils.printTime("Test", testTime);
        DL4J_Utils.printTime("Total", totalTime);

    }

    public static void main(String[] args) throws Exception {
        new BenchmarkCifar().run(args);
    }
}
