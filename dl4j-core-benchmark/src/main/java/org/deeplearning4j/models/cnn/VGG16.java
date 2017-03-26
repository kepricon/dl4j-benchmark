package org.deeplearning4j.models.cnn;

import org.deeplearning4j.models.TestableModel;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 * VGG-16
 */
public class VGG16 implements TestableModel {

    protected static int height;
    protected static int width;
    protected static int channels;
    private int numLabels;
    private long seed;
    private int iterations;

    public VGG16(int height, int width, int channels, int numLabels, long seed, int iterations) {
        this.height = height;
        this.width = width;
        this.channels = channels;
        this.numLabels = numLabels;
        this.seed = seed;
        this.iterations = iterations;
    }

    public MultiLayerConfiguration conf() {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .activation(Activation.RELU)
                .list()
                .layer(0, new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1).padding(1, 1)
                        .nIn(channels)
                        .nOut(64)
                        .cudnnAlgoMode(ConvolutionLayer.AlgoMode.NO_WORKSPACE)
                        .build())
                .layer(1, new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1).padding(1, 1)
                        .nOut(64)
                        .cudnnAlgoMode(ConvolutionLayer.AlgoMode.NO_WORKSPACE)
                        .build())
                .layer(2, new SubsamplingLayer.Builder().poolingType(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                .layer(3, new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1).padding(1, 1)
                        .nOut(128)
                        .cudnnAlgoMode(ConvolutionLayer.AlgoMode.NO_WORKSPACE)
                        .build())
                .layer(4, new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1).padding(1, 1)
                        .nOut(128)
                        .cudnnAlgoMode(ConvolutionLayer.AlgoMode.NO_WORKSPACE)
                        .build())
                .layer(5,
                        new SubsamplingLayer.Builder().poolingType(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2)
                                .stride(2, 2).build())
                .layer(6, new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1).padding(1, 1)
                        .nOut(256)
                        .cudnnAlgoMode(ConvolutionLayer.AlgoMode.NO_WORKSPACE)
                        .build())
                .layer(7, new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1).padding(1, 1)
                        .nOut(256)
                        .cudnnAlgoMode(ConvolutionLayer.AlgoMode.NO_WORKSPACE)
                        .build())
                .layer(8, new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1).padding(1, 1)
                        .nOut(256)
                        .cudnnAlgoMode(ConvolutionLayer.AlgoMode.NO_WORKSPACE)
                        .build())
                .layer(9, new SubsamplingLayer.Builder().poolingType(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                .layer(10, new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1).padding(1, 1)
                        .nOut(512)
                        .cudnnAlgoMode(ConvolutionLayer.AlgoMode.NO_WORKSPACE)
                        .build())
                .layer(11, new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1).padding(1, 1)
                        .nOut(512)
                        .cudnnAlgoMode(ConvolutionLayer.AlgoMode.NO_WORKSPACE)
                        .build())
                .layer(12, new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1).padding(1, 1)
                        .nOut(512)
                        .cudnnAlgoMode(ConvolutionLayer.AlgoMode.NO_WORKSPACE)
                        .build())
                .layer(13, new SubsamplingLayer.Builder().poolingType(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                .layer(14, new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1).padding(1, 1)
                        .nOut(512)
                        .cudnnAlgoMode(ConvolutionLayer.AlgoMode.NO_WORKSPACE)
                        .build())
                .layer(15, new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1).padding(1, 1)
                        .nOut(512)
                        .cudnnAlgoMode(ConvolutionLayer.AlgoMode.NO_WORKSPACE)
                        .build())
                .layer(16, new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1).padding(1, 1)
                        .nOut(512)
                        .cudnnAlgoMode(ConvolutionLayer.AlgoMode.NO_WORKSPACE)
                        .build())
                .layer(17, new SubsamplingLayer.Builder().poolingType(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                .layer(18, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .name("output")
                        .nOut(numLabels)
                        .activation(Activation.SOFTMAX) // radial basis function required
                        .build())
                .backprop(true)
                .pretrain(false)
                .setInputType(InputType.convolutional(height, width, channels))
                .build();

        return conf;
    }

    public MultiLayerNetwork init(){
        MultiLayerNetwork network = new MultiLayerNetwork(conf());
        network.init();
        return network;

    }

}