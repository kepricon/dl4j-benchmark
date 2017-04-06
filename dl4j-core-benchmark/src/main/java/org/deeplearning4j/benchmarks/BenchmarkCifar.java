package org.deeplearning4j.benchmarks;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.ParameterException;
import lombok.extern.slf4j.Slf4j;
import org.datavec.image.loader.CifarLoader;
import org.datavec.image.transform.FlipImageTransform;
import org.datavec.image.transform.ImageTransform;
import org.deeplearning4j.datasets.iterator.impl.CifarDataSetIterator;
import org.deeplearning4j.models.ModelType;
import org.nd4j.jita.conf.CudaEnvironment;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Benchmarks popular CNN models using the CIFAR-10 dataset.
 */
@Slf4j
public class BenchmarkCifar extends BaseBenchmark {

    // values to pass in from command line when compiled, esp running remotely
    @Parameter(names = {"-model","--modelType"}, description = "Model type (e.g. ALEXNET, VGG16, or CNN).")
    public static ModelType modelType = ModelType.ALEXNET;
    @Parameter(names = {"-ng","--numGPUs"}, description = "How many workers to use for multiple GPUs.")
    public int numGPUs = 0;
    @Parameter(names = {"-nTrain","--numTrainExamples"}, description = "Num train examples.")
    public static int numTrainExamples = CifarLoader.NUM_TRAIN_IMAGES; // you can also use
    @Parameter(names = {"-nTrainB","--trainBatchSize"}, description = "Train batch size.")
    public static int trainBatchSize = 125;
    @Parameter(names = {"-pre","--preProcess"}, description = "Set preprocess.")
    public static boolean preProcess = true;
    @Parameter(names = {"-dcache","--deviceCache"}, description = "Set CUDA device cache.")
    public static long deviceCache = 6L;
    @Parameter(names = {"-hcache","--hostCache"}, description = "Set CUDA host cache.")
    public static long hostCache = 12L;
    @Parameter(names = {"-gcthreads","--gcThreads"}, description = "Set Garbage Collection threads.")
    public static int gcThreads = 4;
    @Parameter(names = {"-gcwindow","--gcWindow"}, description = "Set Garbage Collection window in milliseconds.")
    public static int gcWindow = 300;

    protected int height = 224;
    protected int width = 224;
    protected int channels = 3;
    protected int numLabels = CifarLoader.NUM_LABELS;
    protected String datasetName  = "CIFAR-10";
    protected int seed = 42;

    public void run(String[] args) throws Exception {
        // Parse command line arguments if they exist
        JCommander jcmdr = new JCommander(this);
        try {
            jcmdr.parse(args);
        } catch (ParameterException e) {
            //User provides invalid input -> print the usage info
            jcmdr.usage();
            try {
                Thread.sleep(500);
            } catch (Exception e2) {
            }
            System.exit(1);
        }

        // memory management optimizations
        CudaEnvironment.getInstance().getConfiguration()
                .allowMultiGPU(numGPUs > 1 ? true : false)
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

        benchmarkCNN(height, width, channels, numLabels, trainBatchSize, seed, datasetName, cifar, modelType, numGPUs);
    }

    public static void main(String[] args) throws Exception {
        new BenchmarkCifar().run(args);
    }
}
