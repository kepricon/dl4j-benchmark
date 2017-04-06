package org.deeplearning4j.benchmarks;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.ParameterException;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.datasets.MnistDataSetBuilder;
import org.deeplearning4j.datasets.iterator.AsyncDataSetIterator;
import org.deeplearning4j.models.ModelType;
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
public class BenchmarkMnistMLP extends BaseBenchmark {

    public static ModelType modelType = ModelType.SIMPLEMLP;
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

    protected int height = 28;
    protected int width = 28;
    protected int channels = 1;
    protected int numLabels = 10;
    protected String datasetName  = "MNIST";
    protected int seed = 42;
    protected int batch = MnistDataSetBuilder.batchSize;

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

        if (modelType != ModelType.SIMPLEMLP)
            throw new UnsupportedOperationException("Mnist MLP benchmarks are applicable to SIMPLE MLP models only.");

        log.info("Loading data...");
        if(new File(MnistDataSetBuilder.TRAIN_PATH).exists() == false) {
            List<String> dsb_args = new ArrayList<String>();
            dsb_args.add("-b");
            dsb_args.add(String.valueOf(batch));
            dsb_args.add("-s");
            dsb_args.add(String.valueOf(seed));
            new MnistDataSetBuilder().run(dsb_args.toArray(new String[dsb_args.size()]));
        }
        DataSetIterator exsitingTrain = new ExistingMiniBatchDataSetIterator(new File(MnistDataSetBuilder.TRAIN_PATH), "mnist-train-%d.bin");
        DataSetIterator exsitingTest = new ExistingMiniBatchDataSetIterator(new File(MnistDataSetBuilder.TEST_PATH), "mnist-test-%d.bin");
        DataSetIterator train = new AsyncDataSetIterator(exsitingTrain);
        DataSetIterator test = new AsyncDataSetIterator(exsitingTest);

        benchmarkMLP(height, width, channels, numLabels, MnistDataSetBuilder.batchSize, seed, datasetName, train, modelType, numGPUs);
    }

    public static void main(String[] args) throws Exception {
        new BenchmarkMnistMLP().run(args);
    }
}
