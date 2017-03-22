package org.deeplearning4j.ModelCompare.dl4j;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

/**
 *
 */
public class Dl4j_MLP {
    protected static int height;
    protected static int width;
    protected static int channels;
    private int numLabels;
    protected double learningRate;
    protected double momentum;
    protected double l2;
    private long seed;

    public Dl4j_MLP(int height, int width, int channels, int numLabels,
                    double learningRate, double momentum, double l2, long seed) {
        this.height = height;
        this.width = width;
        this.channels = channels;
        this.numLabels = numLabels;
        this.learningRate = learningRate;
        this.momentum = momentum;
        this.l2 = l2;
        this.seed = seed;
    }

    public MultiLayerNetwork build_model() {
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
                .layer(1, new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nIn(hiddenNodes)
                        .nOut(numLabels)
                        .activation(Activation.SOFTMAX)
                        .build())
                .pretrain(false).backprop(true)
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        return model;
    }


}
