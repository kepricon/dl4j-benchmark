package org.deeplearning4j.benchmarks;

import com.beust.jcommander.Parameter;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.io.FilenameUtils;
import org.datavec.api.split.FileSplit;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.TinyImageNetDataSetBuilder;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.callbacks.DataSetDeserializer;
import org.deeplearning4j.datasets.iterator.parallel.FileSplitParallelDataSetIterator;
import org.deeplearning4j.models.ModelType;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.enums.InequalityHandling;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;

import java.io.File;
import java.util.List;
import java.util.Random;

//import org.deeplearning4j.datasets.iterator.ParallelExistingMiniBatchDataSetIterator;

/**
 * Created by kepricon on 17. 3. 31.
 */
@Slf4j
public class BenchmarkTinyImageNet extends BaseBenchmark {
    private static final String DATA_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip";
    private static final String DATA_PATH = FilenameUtils.concat(System.getProperty("java.io.tmpdir"), "tiny-imagenet");
    private static final String DATA_ROOT_DIR = FilenameUtils.concat(DATA_PATH, "tiny-imagenet-200/");
    private static final String LABEL_ID_FILE = DATA_ROOT_DIR + "wnids.txt";
    private static final String LABEL_NAME_FILE = DATA_ROOT_DIR + "words.txt";
    private static final String TRAIN_DIR = DATA_ROOT_DIR + "train/";
    private static final String VALIDATION_DIR = DATA_ROOT_DIR + "val/";
    private static final String VALIDATION_ANNOTATION_FILE = DATA_ROOT_DIR + "val/val_annotations.txt";
    private static final String[] allowedExtensions = new String[]{"JPEG"};


    // values to pass in from command line when compiled, esp running remotely
    @Parameter(names = {"-model","--modelType"}, description = "Model type (e.g. ALEXNET, VGG16, or CNN).")
    public ModelType modelType = ModelType.VGG16;

    @Parameter(names = {"-w","--width"}, description = "Set WIDTH_SIZE.")
    private int width = 224;
    @Parameter(names = {"-h","--height"}, description = "Set HEIGHT_SIZE.")
    private int height = 224;
    @Parameter(names = {"-c","--channel"}, description = "Set CHANNEL_SIZE.")
    private int channels = 3;
    @Parameter(names = {"-b","--batch"}, description = "Set BATCH_SIZE.")
    private int batchSize = 32;

    private int seed = 42;

    public String TRAIN_PATH = FilenameUtils.concat(System.getProperty("java.io.tmpdir"), "dl4j_tinyimagenet_train/");
    public String TEST_PATH = FilenameUtils.concat(System.getProperty("java.io.tmpdir"), "dl4j_tinyimagenet_test/");

    protected void run() throws Exception {
        TRAIN_PATH = TinyImageNetDataSetBuilder.getTrainPath(height, width, batchSize);
        TEST_PATH = TinyImageNetDataSetBuilder.getTestPath(height, width, batchSize);

        if (modelType == ModelType.ALL || modelType == ModelType.RNN)
            throw new UnsupportedOperationException("TinyImageNet benchmarks are applicable to CNN models only.");

        log.info("Loading data...");

//        DataSetIterator train = new ExistingMiniBatchDataSetIterator(new File(TRAIN_PATH));
//        DataSetIterator train = new ParallelExistingMiniBatchDataSetIterator(new File(TRAIN_PATH), numGPUs);
//        train = new AsyncDataSetIterator(train);


//        FileSplitParallelDataSetIterator train = new FileSplitParallelDataSetIterator(new File(TRAIN_PATH), "dataset-%d.bin", new DataSetDeserializer());
        FileSplitParallelDataSetIterator train = new FileSplitParallelDataSetIterator(new File(TRAIN_PATH), "dataset-%d.bin", new DataSetDeserializer(), numGPUs, 10, InequalityHandling.STOP_EVERYONE);

//        Random r = new Random(12345);
//        FileSplit trainSplit = new FileSplit(new File(TRAIN_DIR), allowedExtensions, r);
//        List<String> labelIDs = TinyImageNetDataSetBuilder.loadLabels(LABEL_ID_FILE);
//        ImageRecordReader trainReader = new ImageRecordReader(height,width,channels,new TinyImageNetDataSetBuilder.TrainLabelGenerator(labelIDs));
//        trainReader.initialize(trainSplit);
//        trainReader.setLabels(labelIDs);
//
//        DataSetIterator train = new RecordReaderDataSetIterator(trainReader, batchSize, 1, 200);
//        train.setPreProcessor(new ImagePreProcessingScaler(-1,1,8));

        benchmarkCNN(height, width, channels, TinyImageNetDataSetBuilder.numLabels, batchSize, seed, TinyImageNetDataSetBuilder.DATASETNAME, train, modelType, numGPUs);
    }

    public static void main(String[] args) throws Exception {
        new BenchmarkTinyImageNet().execute(args);
    }
}
