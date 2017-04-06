package org.deeplearning4j.benchmarks;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.ParameterException;
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
    @Parameter(names = {"-model","--modelType"}, description = "Model type (e.g. ALEXNET, VGG16, or CNN).")
    public static ModelType modelType = ModelType.ALEXNET;
    @Parameter(names = {"-dataset","--datasetPath"}, description = "Path to the parent directly of multiple directories of classes of images.")
    public static String datasetPath = null;
    @Parameter(names = {"-labels","--numLabels"}, description = "Num train labels.")
    public static int numLabels = -1;
    @Parameter(names = {"-nTrainB","--trainBatchSize"}, description = "Train batch size.")
    public static int trainBatchSize = 16;

    protected int height = 224;
    protected int width = 224;
    protected int channels = 3;
    protected String datasetName = "CUSTOM";
    protected int seed = 42;

    public void run(String[] args) throws Exception {
        init(args);

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

        benchmarkCNN(height, width, channels, trainRR.getLabels().size(), trainBatchSize, seed, datasetName, iter, modelType, numGPUs);
    }

    public static void main(String[] args) throws Exception {
        new BenchmarkCustom().run(args);
    }
}
