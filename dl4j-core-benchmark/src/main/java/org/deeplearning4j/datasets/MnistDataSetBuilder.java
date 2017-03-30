package org.deeplearning4j.datasets;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.ParameterException;
import org.apache.commons.io.FilenameUtils;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.slf4j.Logger;

import java.io.File;

/**
 * Created by kepricon on 17. 3. 30.
 */
public class MnistDataSetBuilder {
    private static final Logger log = org.slf4j.LoggerFactory.getLogger(MnistDataSetBuilder.class);

    @Parameter(names = {"-b","--batch"}, description = "BatchSize")
    public static int batchSize = 64;

    @Parameter(names = {"-s","--seed"}, description = "seed")
    private int seed = 12345;

    public static final String TRAIN_PATH = FilenameUtils.concat(System.getProperty("java.io.tmpdir"), "dl4j_mnist_train/");
    public static final String TEST_PATH = FilenameUtils.concat(System.getProperty("java.io.tmpdir"), "dl4j_mnist_test/");

    public void run(String[] args) throws Exception {
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

        log.info("Load data....");
        DataSetIterator mnistTrain = new MnistDataSetIterator(batchSize,true,seed);
        DataSetIterator mnistTest = new MnistDataSetIterator(batchSize,false,seed);

        File trainFolder = new File(TRAIN_PATH);
        trainFolder.mkdirs();
        File testFolder = new File(TEST_PATH);
        testFolder.mkdirs();

        log.info("Saving train data to " + trainFolder.getAbsolutePath() +  " and test data to " + testFolder.getAbsolutePath());

        int trainDataSaved = 0;
        int testDataSaved = 0;
        while(mnistTrain.hasNext()) {
            //note that we use trainDataSaved as an index in to which batch this is for the file
            mnistTrain.next().save(new File(trainFolder,"mnist-train-" + trainDataSaved + ".bin"));
            trainDataSaved++;
        }

        while(mnistTest.hasNext()) {
            //note that we use testDataSaved as an index in to which batch this is for the file
            mnistTest.next().save(new File(testFolder,"mnist-test-" + testDataSaved + ".bin"));
            testDataSaved++;
        }

        log.info("Finished pre saving test and train data");
    }
}
