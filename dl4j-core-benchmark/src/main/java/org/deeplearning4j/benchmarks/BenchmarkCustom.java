package org.deeplearning4j.benchmarks;

import lombok.extern.slf4j.Slf4j;
import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.filters.PathFilter;
import org.datavec.api.io.filters.RandomPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.ImageTransform;
import org.datavec.image.transform.ResizeImageTransform;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.models.ModelType;
import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.Option;
import org.nd4j.jita.conf.CudaEnvironment;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.util.Random;

/**
 * Benchmarks popular CNN models using the CIFAR-10 dataset.
 */
@Slf4j
public class BenchmarkCustom extends BaseBenchmark {

    // values to pass in from command line when compiled, esp running remotely
    @Option(name = "--modelType", usage = "Model type (e.g. ALEXNET, VGG16, or CNN).", aliases = "-model")
    public static ModelType modelType = ModelType.ALEXNET;
    //    @Option(name="--numGPUs",usage="How many workers to use for multiple GPUs.",aliases = "-ng")
//    public int numGPUs = 0;
    @Option(name="--datasetPath",usage="Path to the parent directly of multiple directories of classes of images.",aliases = "-dataset")
    public static String datasetPath = null;
    @Option(name="--numLabels",usage="Train batch size.",aliases = "-labels")
    public static int numLabels = -1;
    @Option(name="--trainBatchSize",usage="Train batch size.",aliases = "-batch")
    public static int trainBatchSize = 16;
    @Option(name="--deviceCache",usage="Set CUDA device cache.",aliases = "-dcache")
    public static long deviceCache = 6L;
    @Option(name="--hostCache",usage="Set CUDA host cache.",aliases = "-hcache")
    public static long hostCache = 12L;
    @Option(name="--gcThreads",usage="Set Garbage Collection threads.",aliases = "-gcthreads")
    public static int gcThreads = 5;
    @Option(name="--gcWindow",usage="Set Garbage Collection window in milliseconds.",aliases = "-gcwindow")
    public static int gcWindow = 110;

    protected int height = 224;
    protected int width = 224;
    protected int channels = 3;
    protected String datasetName  = "CUSTOM";
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
                .allowMultiGPU(false)
                .setMaximumDeviceCache(deviceCache * 1024L * 1024L * 1024L)
                .setMaximumHostCache(hostCache * 1024L * 1024L * 1024L)
                .setNumberOfGcThreads(gcThreads);
        Nd4j.create(1);
        Nd4j.getMemoryManager().togglePeriodicGc(true);
        Nd4j.getMemoryManager().setAutoGcWindow(gcWindow);
        Nd4j.getMemoryManager().setOccasionalGcFrequency(0);

        if(modelType == ModelType.ALL || modelType == ModelType.RNN)
            throw new UnsupportedOperationException("Image benchmarks are applicable to CNN models only.");

        if(datasetPath==null)
            throw new IllegalArgumentException("You must specify a valid path to a labelled dataset of images.");

        log.info("Loading data...");
        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
        File mainPath = new File(datasetPath);
        FileSplit fileSplit = new FileSplit(mainPath, NativeImageLoader.ALLOWED_FORMATS, new Random(seed));

        PathFilter pathFilter;
        if(numLabels>-1)
            pathFilter = new BalancedPathFilter(new Random(seed), labelMaker, 1000000, numLabels, 6000);
        else
            pathFilter = new RandomPathFilter(new Random(seed), NativeImageLoader.ALLOWED_FORMATS);

        InputSplit[] split = fileSplit.sample(pathFilter, 1.0);
        ImageTransform resize = new ResizeImageTransform(224, 224);
        RecordReader trainRR = new ImageRecordReader(height, width, channels, labelMaker, resize);
        trainRR.initialize(split[0]);
        RecordReaderDataSetIterator iter = new RecordReaderDataSetIterator(trainRR, trainBatchSize);

        log.info("Preparing benchmarks for "+split[0].locations().length+" images, "+iter.getLabels().size()+" labels");

        benchmark(height, width, channels, trainRR.getLabels().size(), trainBatchSize, seed, datasetName, iter, modelType);
    }

    public static void main(String[] args) throws Exception {
        new BenchmarkCustom().run(args);
    }
}
