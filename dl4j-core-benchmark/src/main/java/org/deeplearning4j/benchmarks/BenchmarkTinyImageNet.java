package org.deeplearning4j.benchmarks;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.ParameterException;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.io.FilenameUtils;
import org.deeplearning4j.models.ModelType;
import org.nd4j.jita.conf.CudaEnvironment;
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
    @Parameter(names = {"-model","--modelType"}, description = "Model type (e.g. ALEXNET, VGG16, or CNN).")
    public static ModelType modelType = ModelType.ALEXNET;
    @Parameter(names = {"-ng","--numGPUs"}, description = "How many workers to use for multiple GPUs.")
    public int numGPUs = 0;
    @Parameter(names = {"-dcache","--deviceCache"}, description = "Set CUDA device cache.")
    public static long deviceCache = 6L;
    @Parameter(names = {"-hcache","--hostCache"}, description = "Set CUDA host cache.")
    public static long hostCache = 12L;
    @Parameter(names = {"-gcthreads","--gcThreads"}, description = "Set Garbage Collection threads.")
    public static int gcThreads = 4;
    @Parameter(names = {"-gcwindow","--gcWindow"}, description = "Set Garbage Collection window in milliseconds.")
    public static int gcWindow = 300;


    @Parameter(names = {"-w","--width"}, description = "Set WIDTH_SIZE.")
    private static int width = 224;
    @Parameter(names = {"-h","--height"}, description = "Set HEIGHT_SIZE.")
    private static int height = 224;
    @Parameter(names = {"-c","--channel"}, description = "Set CHANNEL_SIZE.")
    private static int channels = 3;
    @Parameter(names = {"-b","--batch"}, description = "Set BATCH_SIZE.")
    private static int batchSize = 32;
    protected String datasetName  = "tiny";
    protected int seed = 42;
    private int numLabels = 200;

    public static String TRAIN_PATH = FilenameUtils.concat(System.getProperty("java.io.tmpdir"), "dl4j_tinyimagenet_train/");
    public static String TEST_PATH = FilenameUtils.concat(System.getProperty("java.io.tmpdir"), "dl4j_tinyimagenet_test/");

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

        TRAIN_PATH = FilenameUtils.concat(System.getProperty("java.io.tmpdir"), "dl4j_" +datasetName+"_"+batchSize+"batch_"+height+"x"+width+"/dl4j_tinyimagenet_train/");
        TEST_PATH = FilenameUtils.concat(System.getProperty("java.io.tmpdir"), "dl4j_"+datasetName+"_"+batchSize+"batch_"+height+"x"+width+"/dl4j_tinyimagenet_test/");
        // memory management optimizations
        CudaEnvironment.getInstance().getConfiguration()
                .allowMultiGPU(numGPUs > 1 ? true : false)
                .setMaximumDeviceCache(deviceCache * 1024L * 1024L * 1024L)
                .allowCrossDeviceAccess(true)
                .setMaximumDeviceCacheableLength(32 * 1024 * 1024)
                .setMaximumHostCache(hostCache * 1024L * 1024L * 1024L)
                .setNumberOfGcThreads(gcThreads);

        Nd4j.create(1);
        Nd4j.getMemoryManager().togglePeriodicGc(true);
        Nd4j.getMemoryManager().setAutoGcWindow(gcWindow);
        Nd4j.getMemoryManager().setOccasionalGcFrequency(0);

        if (modelType == ModelType.ALL || modelType == ModelType.RNN)
            throw new UnsupportedOperationException("TinyImageNet benchmarks are applicable to CNN models only.");

        log.info("Loading data...");
        DataSetIterator train = new ExistingMiniBatchDataSetIterator(new File(TRAIN_PATH));
//        train = new AsyncDataSetIterator(train);

        benchmarkCNN(height, width, channels, numLabels, batchSize, seed, datasetName, train, modelType, numGPUs);
    }

    public static void main(String[] args) throws Exception {
        new BenchmarkTinyImageNet().run(args);
    }
}
