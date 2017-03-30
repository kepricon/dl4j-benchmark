package org.deeplearning4j.benchmarks;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.datasets.DataSetsBuilder;
import org.deeplearning4j.models.ModelType;
import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.Option;
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
    @Option(name = "--modelType", usage = "Model type (e.g. W2VSENTIMENT or RNN).", aliases = "-model")
    public static ModelType modelType = ModelType.W2VSENTIMENT;
    @Option(name="--deviceCache",usage="Set CUDA device cache.",aliases = "-dcache")
    public static long deviceCache = 6L;
    @Option(name="--hostCache",usage="Set CUDA host cache.",aliases = "-hcache")
    public static long hostCache = 12L;
    @Option(name="--gcThreads",usage="Set Garbage Collection threads.",aliases = "-gcthreads")
    public static int gcThreads = 4;
    @Option(name="--gcWindow",usage="Set Garbage Collection window in milliseconds.",aliases = "-gcwindow")
    public static int gcWindow = 300;
    @Option(name="--numGPUs",usage="How many workers to use for multiple GPUs. -1 indicates machine's maximum gpus ",aliases = "-ng")
    public static int numGPUs = 0;

    protected String datasetName  = "IMDB review";

    private void run(String[] args) throws Exception {
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

        if(modelType == ModelType.ALL || modelType == ModelType.RNN)
            throw new UnsupportedOperationException("CIFAR-10 benchmarks are applicable to CNN models only.");

        log.info("Loading data...");
//        if(new File(DataSetsBuilder.TRAIN_PATH).exists() == false) {
//            new DataSetsBuilder().run(null);
//        }
        DataSetIterator train = new ExistingMiniBatchDataSetIterator(new File(DataSetsBuilder.TRAIN_PATH));
        DataSetIterator test = new ExistingMiniBatchDataSetIterator(new File(DataSetsBuilder.TEST_PATH));


        benchmarkRNN(datasetName, train, modelType, numGPUs);
    }


    public static void main(String[] args) throws Exception {
        new BenchmarkW2VSentiment().run(args);
    }


}
