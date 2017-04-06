package org.deeplearning4j.benchmarks;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.ParameterException;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.datasets.W2VSentimentDataSetsBuilder;
import org.deeplearning4j.models.ModelType;
import org.nd4j.jita.conf.CudaEnvironment;
import org.nd4j.linalg.dataset.ExistingMiniBatchDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;

/**
 * Created by kepricon on 17. 3. 28.
 */
@Slf4j
public class BenchmarkW2VSentiment extends BaseBenchmark {

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

    protected String datasetName  = "IMDB review";

    private void run(String[] args) throws Exception {
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

        if(modelType == ModelType.ALL || modelType == ModelType.CNN)
            throw new UnsupportedOperationException("W2VSentiment benchmarks are applicable to RNN models only.");

        log.info("Loading data...");
//        if(new File(W2VSentimentDataSetsBuilder.TRAIN_PATH).exists() == false) {
//            new W2VSentimentDataSetsBuilder().run(null);
//        }
        DataSetIterator train = new ExistingMiniBatchDataSetIterator(new File(W2VSentimentDataSetsBuilder.TRAIN_PATH));
        DataSetIterator test = new ExistingMiniBatchDataSetIterator(new File(W2VSentimentDataSetsBuilder.TEST_PATH));


        benchmarkRNN(datasetName, train, modelType, numGPUs);
    }


    public static void main(String[] args) throws Exception {
        new BenchmarkW2VSentiment().run(args);
    }


}
