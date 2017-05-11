package org.deeplearning4j.benchmarks;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.ParameterException;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.io.FilenameUtils;
import org.deeplearning4j.datasets.TinyImageNetDataSetBuilder;
import org.deeplearning4j.datasets.iterator.callbacks.DataSetDeserializer;
import org.deeplearning4j.datasets.iterator.parallel.FileSplitParallelDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.enums.InequalityHandling;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Created by kepricon on 17. 5. 3.
 */
@Slf4j
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
    protected int numGPUs = Nd4j.getAffinityManager().getNumberOfDevices();

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

        FileSplitParallelDataSetIterator train = new FileSplitParallelDataSetIterator(new File(TRAIN_PATH), "dataset-%d.bin", new DataSetDeserializer(), numGPUs, 4, InequalityHandling.STOP_EVERYONE);

        int numDevices = Nd4j.getAffinityManager().getNumberOfDevices();
        AtomicLong cnt = new AtomicLong(0);
        log.info("Calling hasNext() on device_{}", cnt.getAndIncrement() % numDevices);
        while(train.hasNext()) {
            train.next();
            Thread.sleep(100);

            log.info("Calling next on device_{}", cnt.getAndIncrement() % numDevices);
        }
    }

    public static void main(String[] args) throws Exception {
        new RaverTest().exec(args);
    }
}
