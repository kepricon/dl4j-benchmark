package org.deeplearning4j.models.mlp;

import org.deeplearning4j.models.TestableModel;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 * Created by kepricon on 17. 4. 5.
 */
public class SimpleMLP implements TestableModel {
    protected static int height;
    protected static int width;
    protected static int channels;
    private int numLabels;
    protected double learningRate = 6e-3;;
    protected double momentum = 0.9;
    protected double l2 = 1e-4;
    private long seed;

    public SimpleMLP(int height, int width, int channels, int numLabels, long seed) {
        this.height = height;
        this.width = width;
        this.channels = channels;
        this.numLabels = numLabels;
        this.seed = seed;
    }

    public MultiLayerConfiguration conf() {
        int hiddenNodes = 1000;
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .activation(Activation.RELU)
                .weightInit(WeightInit.XAVIER)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .iterations(1)
                .learningRate(learningRate)
                .updater(Updater.NESTEROVS).momentum(momentum)
                .regularization(true).l2(l2)
                .list()
                .layer(0, new DenseLayer.Builder()
                        .nIn(height * width)
                        .nOut(hiddenNodes)
                        .build())
                .layer(1, new DenseLayer.Builder()
                        .nIn(hiddenNodes)
                        .nOut(hiddenNodes)
                        .build())
                .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nIn(hiddenNodes)
                        .nOut(numLabels)
                        .activation(Activation.SOFTMAX)
                        .build())
                .pretrain(false).backprop(true)
                .build();

        return conf;
    }

    @Override
    public Model init() {
        MultiLayerNetwork network = new MultiLayerNetwork(conf());
        network.init();
        return network;
    }
}
