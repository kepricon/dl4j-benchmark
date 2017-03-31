package org.deeplearning4j.benchmarks;

import lombok.extern.slf4j.Slf4j;
import org.datavec.image.loader.CifarLoader;
import org.deeplearning4j.datasets.MnistDataSetBuilder;
import org.deeplearning4j.datasets.iterator.AsyncDataSetIterator;
import org.deeplearning4j.models.ModelType;
import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.Option;
import org.nd4j.jita.conf.CudaEnvironment;
import org.nd4j.linalg.dataset.ExistingMiniBatchDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by kepricon on 17. 3. 30.
 */
@Slf4j
public class BenchmarkMnist extends BaseBenchmark {

    // values to pass in from command line when compiled, esp running remotely
//    @Option(name = "--modelType", usage = "Model type (e.g. LENET, VGG16, or CNN).", aliases = "-model")
    public static ModelType modelType = ModelType.LENET;
    @Option(name="--numGPUs",usage="How many workers to use for multiple GPUs.",aliases = "-ng")
    public int numGPUs = 0;
    @Option(name="--numTrainExamples",usage="Num train examples.",aliases = "-nTrain")
    public static int numTrainExamples = CifarLoader.NUM_TRAIN_IMAGES; // you can also use
//    @Option(name="--trainBatchSize",usage="Train batch size.",aliases = "-nTrainB")
//    public static int trainBatchSize = 125;
    @Option(name="--deviceCache",usage="Set CUDA device cache.",aliases = "-dcache")
    public static long deviceCache = 6L;
    @Option(name="--hostCache",usage="Set CUDA host cache.",aliases = "-hcache")
    public static long hostCache = 12L;
    @Option(name="--gcThreads",usage="Set Garbage Collection threads.",aliases = "-gcthreads")
    public static int gcThreads = 4;
    @Option(name="--gcWindow",usage="Set Garbage Collection window in milliseconds.",aliases = "-gcwindow")
    public static int gcWindow = 300;

    protected int height = 28;
    protected int width = 28;
    protected int channels = 1;
    protected int numLabels = 10;
    protected String datasetName  = "MNIST";
    protected int seed = 42;

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
        CudaEnvironment.getInstance().getConfiguration()
                .allowMultiGPU(numGPUs > 1 ? true : false)
                .setMaximumDeviceCache(deviceCache * 1024L * 1024L * 1024L)
                .setMaximumHostCache(hostCache * 1024L * 1024L * 1024L)
                .setNumberOfGcThreads(gcThreads);
        Nd4j.create(1);
        Nd4j.getMemoryManager().togglePeriodicGc(true);
        Nd4j.getMemoryManager().setAutoGcWindow(gcWindow);
        Nd4j.getMemoryManager().setOccasionalGcFrequency(0);

        if (modelType == ModelType.ALL || modelType == ModelType.RNN)
            throw new UnsupportedOperationException("Mnist benchmarks are applicable to CNN models only.");

        log.info("Loading data...");
        if(new File(MnistDataSetBuilder.TRAIN_PATH).exists() == false) {
            List<String> dsb_args = new ArrayList<String>();
            dsb_args.add("-b");
            dsb_args.add("64");
            dsb_args.add("-s");
            dsb_args.add(String.valueOf(seed));
            new MnistDataSetBuilder().run(dsb_args.toArray(new String[dsb_args.size()]));
        }
        DataSetIterator exsitingTrain = new ExistingMiniBatchDataSetIterator(new File(MnistDataSetBuilder.TRAIN_PATH), "mnist-train-%d.bin");
        DataSetIterator exsitingTest = new ExistingMiniBatchDataSetIterator(new File(MnistDataSetBuilder.TEST_PATH), "mnist-test-%d.bin");
        DataSetIterator train = new AsyncDataSetIterator(exsitingTrain);
        DataSetIterator test = new AsyncDataSetIterator(exsitingTest);

        benchmarkCNN(height, width, channels, numLabels, MnistDataSetBuilder.batchSize, seed, datasetName, train, modelType, numGPUs);
    }

    public static void main(String[] args) throws Exception {
        new BenchmarkMnist().run(args);
    }
}
