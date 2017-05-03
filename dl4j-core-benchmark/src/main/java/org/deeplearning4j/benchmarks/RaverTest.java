package org.deeplearning4j.benchmarks;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.ParameterException;
import org.apache.commons.io.FilenameUtils;
import org.deeplearning4j.datasets.TinyImageNetDataSetBuilder;
import org.deeplearning4j.datasets.iterator.callbacks.DataSetDeserializer;
import org.deeplearning4j.datasets.iterator.parallel.FileSplitParallelDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.enums.InequalityHandling;

import java.io.File;

/**
 * Created by kepricon on 17. 5. 3.
 */
public class RaverTest {
    @Parameter(names = {"-w","--width"}, description = "Set WIDTH_SIZE.")
    private int width = 224;
    @Parameter(names = {"-h","--height"}, description = "Set HEIGHT_SIZE.")
    private int height = 224;
    @Parameter(names = {"-c","--channel"}, description = "Set CHANNEL_SIZE.")
    private int channels = 3;
    @Parameter(names = {"-b","--batch"}, description = "Set BATCH_SIZE.")
    private int batchSize = 32;
    @Parameter(names = {"-ng","--numGPUs"}, description = "How many workers to use for multiple GPUs.")
    protected int numGPUs = 0;

    public String TRAIN_PATH = FilenameUtils.concat(System.getProperty("java.io.tmpdir"), "dl4j_tinyimagenet_train/");
    public String TEST_PATH = FilenameUtils.concat(System.getProperty("java.io.tmpdir"), "dl4j_tinyimagenet_test/");

    public void exec(String[] args) throws Exception {
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

        TRAIN_PATH = TinyImageNetDataSetBuilder.getTrainPath(height, width, batchSize);
        TEST_PATH = TinyImageNetDataSetBuilder.getTestPath(height, width, batchSize);

        FileSplitParallelDataSetIterator train = new FileSplitParallelDataSetIterator(new File(TRAIN_PATH), "dataset-%d.bin", new DataSetDeserializer(), numGPUs, 10, InequalityHandling.STOP_EVERYONE);

        while(train.hasNext()) {
            train.next();
            Thread.sleep(100);
        }
    }

    public static void main(String[] args) throws Exception {
        new RaverTest().exec(args);
    }
}
