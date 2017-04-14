package org.deeplearning4j.benchmarks;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.ParameterException;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.datasets.iterator.MultipleEpochsIterator;
import org.deeplearning4j.listeners.BenchmarkListener;
import org.deeplearning4j.listeners.BenchmarkReport;
import org.deeplearning4j.models.ModelSelector;
import org.deeplearning4j.models.ModelType;
import org.deeplearning4j.models.TestableModel;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.parallelism.ParallelWrapper;
//import org.nd4j.jita.conf.CudaEnvironment;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;

import java.lang.reflect.Method;
import java.util.Map;

/**
 * Benchmarks popular CNN models using the CIFAR-10 dataset.
 */
@Slf4j
public abstract class BaseBenchmark implements Benchmarkable {
    @Parameter(names = {"-ng","--numGPUs"}, description = "How many workers to use for multiple GPUs.")
    protected int numGPUs = 0;
    @Parameter(names = {"-dcache","--deviceCache"}, description = "Set CUDA device cache.")
    protected long deviceCache = 6L;
    @Parameter(names = {"-dcachelength","--deviceCacheLength"}, description = "Set CUDA device cache length.")
    protected long deviceCacheLength = 32L;
    @Parameter(names = {"-hcache","--hostCache"}, description = "Set CUDA host cache.")
    protected long hostCache = 12L;
    @Parameter(names = {"-gcthreads","--gcThreads"}, description = "Set Garbage Collection threads.")
    protected int gcThreads = 4;
    @Parameter(names = {"-gcwindow","--gcWindow"}, description = "Set Garbage Collection window in milliseconds.")
    protected int gcWindow = 300;
    @Parameter(names = {"-miter","--maxIteration"}, description = "Max Iteration. -1 means no limit")
    protected long maxIteration = -1;

    protected int listenerFreq = 10;
    protected int iterations = 1;
    protected static Map<ModelType,TestableModel> networks;


    @Override
    public void execute(String[] args) throws Exception {
        init(args);
        run();
    }

    protected abstract void run() throws Exception;

    protected void init(String[] args){
        // Parse command line arguments if they exist
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

        // memory management optimizations
//        CudaEnvironment
//                .getInstance()
//                .getConfiguration()
//                .allowMultiGPU(numGPUs > 1 ? true : false)
//                .setMaximumDeviceCache(deviceCache * 1024L * 1024L * 1024L)
//                .allowCrossDeviceAccess(true)
//                .setMaximumDeviceCacheableLength(deviceCacheLength * 1024L * 1024L)
//                .setMaximumHostCache(hostCache * 1024L * 1024L * 1024L)
//                .setNumberOfGcThreads(gcThreads);
//
        Nd4j.create(1);
        Nd4j.getMemoryManager().togglePeriodicGc(false);
        Nd4j.getMemoryManager().setAutoGcWindow(gcWindow);
        Nd4j.getMemoryManager().setOccasionalGcFrequency(0);
    }

    public void benchmarkCNN(int height, int width, int channels, int numLabels, int batchSize, int seed, String datasetName, DataSetIterator iter, ModelType modelType, int numGPUs) throws Exception {
        log.info("Building models for "+modelType+"....");
        networks = ModelSelector.select(modelType, height, width, channels, numLabels, seed, iterations);
        String dimensions = datasetName+" "+batchSize+"x"+channels+"x"+height+"x"+width;

        benchmark(dimensions, iter, numGPUs);
    }

    public void benchmarkRNN(String datasetName, DataSetIterator iter, ModelType modelType, int numGPUs) throws Exception {
        log.info("Building models for " + modelType + "....");
        networks = ModelSelector.select(modelType, 0, 0, 0, 0, 0, 0);
        String dimensions = datasetName;

        benchmark(dimensions, iter, numGPUs);
    }

    public void benchmarkMLP(int height, int width, int channels, int numLabels, int batchSize, int seed, String datasetName, DataSetIterator iter, ModelType modelType, int numGPUs) throws Exception {
        log.info("Building models for "+modelType+"....");
        networks = ModelSelector.select(modelType, height, width, channels, numLabels, seed, iterations);
        String dimensions = datasetName+" "+batchSize+"x"+channels+"x"+height+"x"+width;

        benchmark(dimensions, iter, numGPUs);
    }

    private void benchmark(String description, DataSetIterator iter, int numGPUs) throws Exception {
        long totalTime = System.currentTimeMillis();

        if (maxIteration != -1){
            iter = new MultipleEpochsIterator(iter, maxIteration);
        }

        log.info("========================================");
        log.info("===== Benchmarking selected models =====");
        log.info("========================================");

        for (Map.Entry<ModelType, TestableModel> net : networks.entrySet()) {

            log.info("Selected: "+net.getKey().toString()+" "+description);
            Model model = net.getValue().init();
//            BenchmarkReport report = new BenchmarkReport(net.getKey().toString(), description);
            BenchmarkReport report = BenchmarkReport.getInstance();
            report.setName(net.getKey().toString());
            report.setDescription(description);
            report.setModel(model);

            model.setListeners(new ScoreIterationListener(listenerFreq), new BenchmarkListener(report));

            long epochTime = System.currentTimeMillis();
            log.info("===== Benchmarking training iteration =====");
            if (numGPUs == 0 || numGPUs == 1) { // cpu mode or single gpu mode
                if (model instanceof MultiLayerNetwork) {
                    ((MultiLayerNetwork) model).fit(iter);
                }else if (model instanceof ComputationGraph) {
                    ((ComputationGraph) model).fit(iter);
                }
            } else { // multiple gpu mode
                numGPUs = (numGPUs == -1) ? Nd4j.getAffinityManager().getNumberOfDevices() : numGPUs;
                ParallelWrapper pw = new ParallelWrapper.Builder<>(model)
                        .prefetchBuffer(numGPUs)
                        .reportScoreAfterAveraging(true)
                        .averagingFrequency(10)
                        .useLegacyAveraging(false)
                        .useMQ(true)
                        .workers(numGPUs)
                        .averageUpdaters(false)
                        .build();

                pw.fit(iter);
                pw.close();
            }
            epochTime = System.currentTimeMillis() - epochTime;
            report.setEpochTime(epochTime);


            log.info("===== Benchmarking forward/backward pass =====");
            /*
                Notes: popular benchmarks will measure the time it takes to set the input and feed forward
                and backward. This is consistent with benchmarks seen in the wild like this code:
                https://github.com/jcjohnson/cnn-benchmarks/blob/master/cnn_benchmark.lua
             */
            calcFwdBwdTime(model, iter, report);

            log.info("=============================");
            log.info("===== Benchmark Results =====");
            log.info("=============================");

            System.out.println(report.getModelSummary());
            System.out.println(report.toString());
        }
    }

    private void calcFwdBwdTime(Model model, DataSetIterator iter, BenchmarkReport report) throws Exception {
        iter.reset(); // prevents NPE
        long totalForward = 0;
        long totalBackward = 0;
        long nIterations = 0;

        while(iter.hasNext()) {
            DataSet ds = iter.next();
            INDArray input = ds.getFeatures();
            INDArray labels = ds.getLabels();

            // forward
            long forwardTime = System.currentTimeMillis();
            if(model instanceof MultiLayerNetwork){
                ((MultiLayerNetwork) model).setInput(input);
                ((MultiLayerNetwork) model).setLabels(labels);
                ((MultiLayerNetwork) model).feedForward();
            }else if(model instanceof ComputationGraph){
                ((ComputationGraph) model).setInput(0, input);
                ((ComputationGraph) model).setLabel(0, labels);
                ((ComputationGraph) model).feedForward();
            }
            forwardTime = System.currentTimeMillis() - forwardTime;
            totalForward += forwardTime;

            // backward
            long backwardTime = System.currentTimeMillis();
            if(model instanceof MultiLayerNetwork){
                Method m = MultiLayerNetwork.class.getDeclaredMethod("backprop"); // requires reflection
                m.setAccessible(true);
                m.invoke(model);
            }else if(model instanceof ComputationGraph){
                Method m = ComputationGraph.class.getDeclaredMethod("calcBackpropGradients", boolean.class, INDArray[].class);
                m.setAccessible(true);
                m.invoke(model, false, new INDArray[0]);
            }

            backwardTime = System.currentTimeMillis() - backwardTime;
            totalBackward += backwardTime;

            nIterations += 1;
            if(nIterations % 100 == 0) log.info("Completed "+nIterations+" iterations");
        }

        report.setAvgFeedForward((double) totalForward / (double) nIterations);
        report.setAvgBackprop((double) totalBackward / (double) nIterations);
    }
}
