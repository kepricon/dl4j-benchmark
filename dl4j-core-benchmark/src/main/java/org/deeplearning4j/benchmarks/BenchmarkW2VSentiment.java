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

    protected String datasetName  = "IMDB review";

    private void run(String[] args) throws Exception {
        init(args);

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
