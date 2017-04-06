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
    public ModelType modelType = ModelType.ALEXNET;

    @Parameter(names = {"-w","--width"}, description = "Set WIDTH_SIZE.")
    private int width = 224;
    @Parameter(names = {"-h","--height"}, description = "Set HEIGHT_SIZE.")
    private int height = 224;
    @Parameter(names = {"-c","--channel"}, description = "Set CHANNEL_SIZE.")
    private int channels = 3;
    @Parameter(names = {"-b","--batch"}, description = "Set BATCH_SIZE.")
    private int batchSize = 32;

    protected String datasetName  = "tiny";
    protected int seed = 42;
    private int numLabels = 200;

    public String TRAIN_PATH = FilenameUtils.concat(System.getProperty("java.io.tmpdir"), "dl4j_tinyimagenet_train/");
    public String TEST_PATH = FilenameUtils.concat(System.getProperty("java.io.tmpdir"), "dl4j_tinyimagenet_test/");

    public void run(String[] args) throws Exception {
        init(args);

        TRAIN_PATH = FilenameUtils.concat(System.getProperty("java.io.tmpdir"), "dl4j_" +datasetName+"_"+batchSize+"batch_"+height+"x"+width+"/dl4j_tinyimagenet_train/");
        TEST_PATH = FilenameUtils.concat(System.getProperty("java.io.tmpdir"), "dl4j_"+datasetName+"_"+batchSize+"batch_"+height+"x"+width+"/dl4j_tinyimagenet_test/");

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
