package org.deeplearning4j.benchmarks;

import lombok.extern.slf4j.Slf4j;
import org.datavec.image.loader.CifarLoader;
import org.datavec.image.transform.FlipImageTransform;
import org.datavec.image.transform.ImageTransform;
import org.deeplearning4j.datasets.iterator.impl.CifarDataSetIterator;
import org.deeplearning4j.models.ModelType;
import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.Option;
import org.nd4j.jita.conf.CudaEnvironment;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Benchmarks popular CNN models using the CIFAR-10 dataset.
 */
@Slf4j
public class BenchmarkCifar extends BaseBenchmark {

    // values to pass in from command line when compiled, esp running remotely
    @Option(name = "--modelType", usage = "Model type (e.g. ALEXNET, VGG16, or CNN).", aliases = "-model")
    public static ModelType modelType = ModelType.ALEXNET;
    //    @Option(name="--numGPUs",usage="How many workers to use for multiple GPUs.",aliases = "-ng")
//    public int numGPUs = 0;
    @Option(name="--numTrainExamples",usage="Num train examples.",aliases = "-nTrain")
    public static int numTrainExamples = CifarLoader.NUM_TRAIN_IMAGES;
    @Option(name="--trainBatchSize",usage="Train batch size.",aliases = "-nTrainB")
    public static int trainBatchSize = 125;
    @Option(name="--preProcess",usage="Set preprocess.",aliases = "-pre")
    public static boolean preProcess = true;
    @Option(name="--deviceCache",usage="Set CUDA device cache.",aliases = "-dcache")
    public static long deviceCache = 6L;
    @Option(name="--hostCache",usage="Set CUDA host cache.",aliases = "-hcache")
    public static long hostCache = 12L;
    @Option(name="--gcThreads",usage="Set Garbage Collection threads.",aliases = "-gcthreads")
    public static int gcThreads = 4;
    @Option(name="--gcWindow",usage="Set Garbage Collection window in milliseconds.",aliases = "-gcwindow")
    public static int gcWindow = 300;



    protected int height = 224;
    protected int width = 224;
    protected int channels = 3;
    protected int numLabels = CifarLoader.NUM_LABELS;
    protected String datasetName  = "CIFAR-10";
    protected int seed = 42;

    public void run(String[] args) {
        // Parse command line arguments if they exist
        CmdLineParser parser = new CmdLineParser(this);
        try {
            parser.parseArgument(args);
        } catch (CmdLineException e) {
            // handling of wrong arguments
            System.err.println(e.getMessage());
            parser.printUsage(System.err);
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

        if(modelType == ModelType.ALL || modelType == ModelType.RNN)
            throw new UnsupportedOperationException("CIFAR-10 benchmarks are applicable to CNN models only.");

        log.info("Loading data...");
        ImageTransform flip = new FlipImageTransform(seed); // Should random flip some images but not all
        DataSetIterator cifar = new CifarDataSetIterator(trainBatchSize, numTrainExamples, new int[]{height, width, channels}, numLabels, flip, preProcess, train);

        benchmark(height, width, channels, numLabels, trainBatchSize, seed, datasetName, cifar, modelType);
    }

    public static void main(String[] args) throws Exception {
        new BenchmarkCifar().run(args);
    }
}
