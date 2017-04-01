package org.deeplearning4j.models.cnn;

import org.deeplearning4j.models.TestableModel;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.distribution.GaussianDistribution;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.lossfunctions.LossFunctions;


/**
 * AlexNet
 *
 * Dl4j's AlexNet model interpretation based on the original paper ImageNet Classification with Deep Convolutional Neural Networks
 * and the imagenetExample code referenced.
 *
 * References:
 * http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
 * https://github.com/BVLC/caffe/blob/master/models/bvlc_alexnet/train_val.prototxt
 *
 * Model is built in dl4j based on available functionality and notes indicate where there are gaps waiting for enhancements.
 *
 * Bias initialization in the paper is 1 in certain layers but 0.1 in the imagenetExample code
 * Weight distribution uses 0.1 std for all layers in the paper but 0.005 in the dense layers in the imagenetExample code
 *
 */
public class AlexNet implements TestableModel {

    private int height;
    private int width;
    private int channels;
    private int numLabels = 1000;
    private long seed = 42;
    private int iterations = 90;

    public AlexNet(int height, int width, int channels, int numLabels, long seed, int iterations) {
        this.height = height;
        this.width = width;
        this.channels = channels;
        this.numLabels = numLabels;
        this.seed = seed;
        this.iterations = iterations;
    }

    public MultiLayerConfiguration conf() {
//        double nonZeroBias = 1;
//        double dropOut = 0.5;
//
//        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
//                .seed(seed)
//                .weightInit(WeightInit.DISTRIBUTION)
//                .dist(new NormalDistribution(0.0, 0.01))
//                .activation(Activation.RELU)
//                .updater(Updater.NESTEROVS)
//                .iterations(iterations)
//                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer) // normalize to prevent vanishing or exploding gradients
//                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
//                .learningRate(1e-2)
//                .momentum(0.9)
//                .biasLearningRate(1e-2*2)
//                .learningRateDecayPolicy(LearningRatePolicy.Step)
//                .lrPolicyDecayRate(0.1)
//                .lrPolicySteps(100000)
//                .regularization(true)
////                .convolutionMode(ConvolutionMode.Same)
//                .convolutionMode(ConvolutionMode.Truncate)
//                .dropOut(0.5)
//                .l2(5 * 1e-4)
//                .miniBatch(false)
//                .list()
//                .layer(0, new ConvolutionLayer.Builder(new int[]{11,11}, new int[]{4, 4}, new int[]{2,2})
//                        .name("cnn1")
//                        .cudnnAlgoMode(ConvolutionLayer.AlgoMode.NO_WORKSPACE)
////                        .convolutionMode(ConvolutionMode.Truncate)
//                        .nIn(channels)
//                        .nOut(64)
//                        .build())
//                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{3,3}, new int[]{2,2})
//                        .name("maxpool1")
//                        .build())
////                .layer(2, new ConvolutionLayer.Builder(new int[]{5,5}, new int[]{2,2}, new int[]{2,2}) // TODO: fix input and put stride back to 1,1
//                .layer(2, new ConvolutionLayer.Builder(new int[]{5,5}, new int[]{1,1}, new int[]{2,2}) // TODO: fix input and put stride back to 1,1
//                        .name("cnn2")
//                        .cudnnAlgoMode(ConvolutionLayer.AlgoMode.NO_WORKSPACE)
//                        .nOut(192)
//                        .biasInit(nonZeroBias)
//                        .build())
//                .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{3, 3}, new int[]{2, 2})
//                        .name("maxpool2")
//                        .build())
//                .layer(4, new ConvolutionLayer.Builder(new int[]{3,3}, new int[]{1,1}, new int[]{1,1})
//                        .name("cnn3")
//                        .cudnnAlgoMode(ConvolutionLayer.AlgoMode.NO_WORKSPACE)
//                        .nOut(384)
//                        .build())
//                .layer(5, new ConvolutionLayer.Builder(new int[]{3,3}, new int[]{1,1}, new int[]{1,1})
//                        .name("cnn4")
//                        .cudnnAlgoMode(ConvolutionLayer.AlgoMode.NO_WORKSPACE)
//                        .nOut(256)
//                        .biasInit(nonZeroBias)
//                        .build())
//                .layer(6, new ConvolutionLayer.Builder(new int[]{3,3}, new int[]{1,1}, new int[]{1,1})
//                        .name("cnn5")
//                        .cudnnAlgoMode(ConvolutionLayer.AlgoMode.NO_WORKSPACE)
//                        .nOut(256)
//                        .biasInit(nonZeroBias)
//                        .build())
////                .layer(7, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{3,3}, new int[]{7,7}) // TODO: fix input and put stride back to 2,2
//                .layer(7, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{3,3}, new int[]{2,2}) // TODO: fix input and put stride back to 2,2
//                        .name("maxpool3")
//                        .build())
//                .layer(8, new DenseLayer.Builder()
//                        .name("ffn1")
//                        .nIn(256)
//                        .nOut(4096)
//                        .dist(new GaussianDistribution(0, 0.005))
//                        .biasInit(nonZeroBias)
//                        .dropOut(dropOut)
//                        .build())
//                .layer(9, new DenseLayer.Builder()
//                        .name("ffn2")
//                        .nOut(4096)
//                        .dist(new GaussianDistribution(0, 0.005))
//                        .biasInit(nonZeroBias)
//                        .dropOut(dropOut)
//                        .build())
//                .layer(10, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
//                        .name("output")
//                        .nOut(numLabels)
//                        .activation(Activation.SOFTMAX)
//                        .build())
//                .backprop(true)
//                .pretrain(false)
//                .setInputType(InputType.convolutional(height,width,channels))
//                .build();

        double nonZeroBias = 1;
        double dropOut = 0.5;
        SubsamplingLayer.PoolingType poolingType = SubsamplingLayer.PoolingType.MAX;

        // TODO split and link kernel maps on GPUs - 2nd, 4th, 5th convolution should only connect maps on the same gpu, 3rd connects to all in 2nd
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .weightInit(WeightInit.DISTRIBUTION)
                .dist(new NormalDistribution(0.0, 0.01))
                .activation("relu")
                .updater(Updater.NESTEROVS)
                .iterations(iterations)
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer) // normalize to prevent vanishing or exploding gradients
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .learningRate(1e-2)
                .biasLearningRate(1e-2*2)
                .learningRateDecayPolicy(LearningRatePolicy.Step)
                .lrPolicyDecayRate(0.1)
                .lrPolicySteps(100000)
                .regularization(true)
                .l2(5 * 1e-4)
                .momentum(0.9)
                .miniBatch(false)
                .list()
                .layer(0, new ConvolutionLayer.Builder(new int[]{11, 11}, new int[]{4, 4}, new int[]{3, 3})
                        .name("cnn1")
                        .nIn(channels)
                        .nOut(96)
                        .build())
                .layer(1, new LocalResponseNormalization.Builder()
                        .name("lrn1")
                        .build())
                .layer(2, new SubsamplingLayer.Builder(poolingType, new int[]{3, 3}, new int[]{2, 2})
                        .name("maxpool1")
                        .build())
                .layer(3, new ConvolutionLayer.Builder(new int[]{5, 5}, new int[]{1, 1}, new int[]{2, 2})
                        .name("cnn2")
                        .nOut(256)
                        .biasInit(nonZeroBias)
                        .build())
                .layer(4, new LocalResponseNormalization.Builder()
                        .name("lrn2")
                        .k(2).n(5).alpha(1e-4).beta(0.75)
                        .build())
                .layer(5, new SubsamplingLayer.Builder(poolingType, new int[]{3, 3}, new int[]{2, 2})
                        .name("maxpool2")
                        .build())
                .layer(6, new ConvolutionLayer.Builder(new int[]{3, 3}, new int[]{1, 1}, new int[]{1, 1})
                        .name("cnn3")
                        .nOut(384)
                        .build())
                .layer(7, new ConvolutionLayer.Builder(new int[]{3, 3}, new int[]{1, 1}, new int[]{1, 1})
                        .name("cnn4")
                        .nOut(384)
                        .biasInit(nonZeroBias)
                        .build())
                .layer(8, new ConvolutionLayer.Builder(new int[]{3, 3}, new int[]{1, 1}, new int[]{1, 1})
                        .name("cnn5")
                        .nOut(256)
                        .biasInit(nonZeroBias)
                        .build())
                .layer(9, new SubsamplingLayer.Builder(poolingType, new int[]{3, 3}, new int[]{2, 2})
                        .name("maxpool3")
                        .build())
                .layer(10, new DenseLayer.Builder()
                        .name("ffn1")
                        .nOut(4096)
                        .dist(new GaussianDistribution(0, 0.005))
                        .biasInit(nonZeroBias)
                        .dropOut(dropOut)
                        .build())
                .layer(11, new DenseLayer.Builder()
                        .name("ffn2")
                        .nOut(4096)
                        .dist(new GaussianDistribution(0, 0.005))
                        .biasInit(nonZeroBias)
                        .dropOut(dropOut)
                        .build())
                .layer(12, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .name("output")
                        .nOut(numLabels)
                        .activation("softmax")
                        .build())
                .backprop(true)
                .pretrain(false)
                .setInputType(InputType.convolutional(height,width,channels)).build();

        return conf;
    }

    public MultiLayerNetwork init(){
        MultiLayerNetwork network = new MultiLayerNetwork(conf());
        network.init();
        return network;

    }

}