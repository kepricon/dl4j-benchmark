package org.deeplearning4j.benchmarks;

import lombok.extern.slf4j.Slf4j;
import org.apache.commons.io.FilenameUtils;
import org.datavec.image.loader.CifarLoader;
import org.deeplearning4j.datasets.iterator.AsyncDataSetIterator;
import org.deeplearning4j.models.ModelType;
import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.Option;
//import org.nd4j.jita.conf.CudaEnvironment;
import org.nd4j.linalg.dataset.ExistingMiniBatchDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;

/**
 * Created by kepricon on 17. 3. 31.
 */
@Slf4j
public class BenchmarkTinyImageNet extends BaseBenchmark {
    // values to pass in from command line when compiled, esp running remotely
    @Option(name = "--modelType", usage = "Model type (e.g. ALEXNET, VGG16, or CNN).", aliases = "-model")
    public static ModelType modelType = ModelType.ALEXNET;
    @Option(name="--numGPUs",usage="How many workers to use for multiple GPUs.",aliases = "-ng")
    public int numGPUs = 0;
    @Option(name="--numTrainExamples",usage="Num train examples.",aliases = "-nTrain")
    public static int numTrainExamples = CifarLoader.NUM_TRAIN_IMAGES; // you can also use
    @Option(name="--trainBatchSize",usage="Train batch size.",aliases = "-nTrainB")
    public static int trainBatchSize = 64;
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
    protected int numLabels = 200;
    protected String datasetName  = "TinyImageNet";
    protected int seed = 42;

    public static final String TRAIN_PATH = FilenameUtils.concat(System.getProperty("java.io.tmpdir"), "dl4j_tinyimagenet_train/");
    public static final String TEST_PATH = FilenameUtils.concat(System.getProperty("java.io.tmpdir"), "dl4j_tinyimagenet_test/");

    public void run(String[] args) throws Exception {
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
//        CudaEnvironment.getInstance().getConfiguration()
//                .allowMultiGPU(numGPUs > 1 ? true : false)
//                .setMaximumDeviceCache(deviceCache * 1024L * 1024L * 1024L)
//                .setMaximumHostCache(hostCache * 1024L * 1024L * 1024L)
//                .setNumberOfGcThreads(gcThreads);
//        Nd4j.create(1);
//        Nd4j.getMemoryManager().togglePeriodicGc(true);
//        Nd4j.getMemoryManager().setAutoGcWindow(gcWindow);
//        Nd4j.getMemoryManager().setOccasionalGcFrequency(0);

        if (modelType == ModelType.ALL || modelType == ModelType.RNN)
            throw new UnsupportedOperationException("TinyImageNet benchmarks are applicable to CNN models only.");

        log.info("Loading data...");
        DataSetIterator train = new ExistingMiniBatchDataSetIterator(new File(TRAIN_PATH));
//        train = new AsyncDataSetIterator(train);

        benchmarkCNN(height, width, channels, numLabels, trainBatchSize, seed, datasetName, train, modelType, numGPUs);
    }

    public static void main(String[] args) throws Exception {
        new BenchmarkTinyImageNet().run(args);
    }
}
