package org.deeplearning4j.models.cnn;

import org.deeplearning4j.models.TestableModel;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
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
                .trainingWorkspaceMode(WorkspaceMode.SEPARATE)
                .inferenceWorkspaceMode(WorkspaceMode.SEPARATE)
                .seed(seed)
                .activation(Activation.RELU)
                .updater(Updater.NESTEROVS)
                // TODO pretrain with smaller net for first couple CNN layer weights, use Distribution for rest OR http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf with Relu
                .weightInit(WeightInit.RELU)
                //                .dist(new NormalDistribution(0.0, 0.01)) // uncomment if using WeightInit.DISTRIBUTION
                .iterations(iterations)
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer) // normalize to prevent vanishing or exploding gradients
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .learningRate(1e-1)
                .learningRateScoreBasedDecayRate(1e-1)
                .regularization(true)
                .l2(5 * 1e-4)
                .momentum(0.9)
                .list()
                .layer(0, new ConvolutionLayer.Builder(new int[]{3, 3}, new int[]{1, 1}, new int[]{1, 1})
                        .name("cnn1")
                        .nIn(channels)
                        .nOut(64)
                        .cudnnAlgoMode(ConvolutionLayer.AlgoMode.NO_WORKSPACE)
                        .build())
                .layer(1, new ConvolutionLayer.Builder(new int[]{3, 3}, new int[]{1, 1}, new int[]{1, 1})
                        .name("cnn2")
                        .nOut(64)
                        .cudnnAlgoMode(ConvolutionLayer.AlgoMode.NO_WORKSPACE)
                        .build())
                .layer(2, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{2, 2})
                        .name("maxpool1")
                        .build())
                .layer(3, new ConvolutionLayer.Builder(new int[]{3, 3}, new int[]{1, 1}, new int[]{1, 1})
                        .name("cnn3")
                        .nOut(128)
                        .cudnnAlgoMode(ConvolutionLayer.AlgoMode.NO_WORKSPACE)
                        .build())
                .layer(4, new ConvolutionLayer.Builder(new int[]{3, 3}, new int[]{1, 1}, new int[]{1, 1})
                        .name("cnn4")
                        .nOut(128)
                        .cudnnAlgoMode(ConvolutionLayer.AlgoMode.NO_WORKSPACE)
                        .build())
                .layer(5, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{2, 2})
                        .name("maxpool2")
                        .build())
                .layer(6, new ConvolutionLayer.Builder(new int[]{3, 3}, new int[]{1, 1}, new int[]{1, 1})
                        .name("cnn5")
                        .nOut(256)
                        .cudnnAlgoMode(ConvolutionLayer.AlgoMode.NO_WORKSPACE)
                        .build())
                .layer(7, new ConvolutionLayer.Builder(new int[]{3, 3}, new int[]{1, 1}, new int[]{1, 1})
                        .name("cnn6")
                        .nOut(256)
                        .cudnnAlgoMode(ConvolutionLayer.AlgoMode.NO_WORKSPACE)
                        .build())
                .layer(8, new ConvolutionLayer.Builder(new int[]{3, 3}, new int[]{1, 1}, new int[]{1, 1})
                        .name("cnn7")
                        .nOut(256)
                        .cudnnAlgoMode(ConvolutionLayer.AlgoMode.NO_WORKSPACE)
                        .build())
                .layer(9, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{2, 2})
                        .name("maxpool3")
                        .build())
                .layer(10, new ConvolutionLayer.Builder(new int[]{3, 3}, new int[]{1, 1}, new int[]{1, 1})
                        .name("cnn8")
                        .nOut(512)
                        .cudnnAlgoMode(ConvolutionLayer.AlgoMode.NO_WORKSPACE)
                        .build())
                .layer(11, new ConvolutionLayer.Builder(new int[]{3, 3}, new int[]{1, 1}, new int[]{1, 1})
                        .name("cnn9")
                        .nOut(512)
                        .cudnnAlgoMode(ConvolutionLayer.AlgoMode.NO_WORKSPACE)
                        .build())
                .layer(12, new ConvolutionLayer.Builder(new int[]{3, 3}, new int[]{1, 1}, new int[]{1, 1})
                        .name("cnn10")
                        .nOut(512)
                        .cudnnAlgoMode(ConvolutionLayer.AlgoMode.NO_WORKSPACE)
                        .build())
                .layer(13, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{2, 2})
                        .name("maxpool4")
                        .build())
                .layer(14, new ConvolutionLayer.Builder(new int[]{3, 3}, new int[]{1, 1}, new int[]{1, 1})
                        .name("cnn11")
                        .nOut(512)
                        .cudnnAlgoMode(ConvolutionLayer.AlgoMode.NO_WORKSPACE)
                        .build())
                .layer(15, new ConvolutionLayer.Builder(new int[]{3, 3}, new int[]{1, 1}, new int[]{1, 1})
                        .name("cnn12")
                        .nOut(512)
                        .cudnnAlgoMode(ConvolutionLayer.AlgoMode.NO_WORKSPACE)
                        .build())
                .layer(16, new ConvolutionLayer.Builder(new int[]{3, 3}, new int[]{1, 1}, new int[]{1, 1})
                        .name("cnn13")
                        .nOut(512)
                        .cudnnAlgoMode(ConvolutionLayer.AlgoMode.NO_WORKSPACE)
                        .build())
                .layer(17, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{2, 2})
                        .name("maxpool5")
                        .build())
                .layer(18, new DenseLayer.Builder()
                        .name("ffn1")
                        .nOut(4096)
                        .dropOut(0.5)
                        .build())
                .layer(19, new DenseLayer.Builder()
                        .name("ffn2")
                        .nOut(4096)
                        .dropOut(0.5)
                        .build())
                .layer(20, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .name("output")
                        .nOut(numLabels)
                        .activation(Activation.SOFTMAX)
                        .build())
                .backprop(true)
                .pretrain(false)
                .setInputType(InputType.convolutional(height, width, channels)).build();

        return conf;
    }

    public MultiLayerNetwork init(){
        MultiLayerNetwork network = new MultiLayerNetwork(conf());
        network.init();
        return network;

    }

}