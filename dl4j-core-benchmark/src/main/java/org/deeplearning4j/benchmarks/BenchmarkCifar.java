package org.deeplearning4j.benchmarks;

import com.beust.jcommander.Parameter;
import lombok.extern.slf4j.Slf4j;
import org.datavec.image.loader.CifarLoader;
import org.datavec.image.transform.FlipImageTransform;
import org.datavec.image.transform.ImageTransform;
import org.deeplearning4j.datasets.iterator.impl.CifarDataSetIterator;
import org.deeplearning4j.models.ModelType;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

//import org.nd4j.jita.conf.CudaEnvironment;

/**
 * Benchmarks popular CNN models using the CIFAR-10 dataset.
 */
@Slf4j
public class BenchmarkCifar extends BaseBenchmark {

    // values to pass in from command line when compiled, esp running remotely
    @Parameter(names = {"-model","--modelType"}, description = "Model type (e.g. ALEXNET, VGG16, or CNN).")
    public static ModelType modelType = ModelType.ALEXNET;

    @Parameter(names = {"-nTrain","--numTrainExamples"}, description = "Num train examples.")
    public static int numTrainExamples = CifarLoader.NUM_TRAIN_IMAGES; // you can also use
    @Parameter(names = {"-nTrainB","--trainBatchSize"}, description = "Train batch size.")
    public static int trainBatchSize = 125;
    @Parameter(names = {"-pre","--preProcess"}, description = "Set preprocess.")
    public static boolean preProcess = true;

    protected boolean train = true;

    protected int height = 224;
    protected int width = 224;
    protected int channels = 3;
    protected int numLabels = CifarLoader.NUM_LABELS;
    protected String datasetName  = "CIFAR-10";
    protected int seed = 42;

    protected void run() throws Exception {
        if(modelType == ModelType.ALL || modelType == ModelType.RNN)
            throw new UnsupportedOperationException("CIFAR-10 benchmarks are applicable to CNN models only.");

        log.info("Loading data...");
        ImageTransform flip = new FlipImageTransform(seed); // Should random flip some images but not all
        DataSetIterator cifar = new CifarDataSetIterator(trainBatchSize, numTrainExamples, new int[]{height, width, channels}, numLabels, flip, preProcess, train);

        benchmarkCNN(height, width, channels, numLabels, trainBatchSize, seed, datasetName, cifar, modelType, numGPUs);
    }

    public static void main(String[] args) throws Exception {
        new BenchmarkCifar().execute(args);
    }
}
