package org.deeplearning4j.models.rnn;

import org.deeplearning4j.models.TestableModel;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.PerformanceListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 * * This example is almost identical to the GravesLSTMCharModellingExample, except that it utilizes the ComputationGraph
 * architecture instead of MultiLayerNetwork architecture. See the javadoc in that example for details.
 * For more details on the ComputationGraph architecture, see http://deeplearning4j.org/compgraph
 *
 * In addition to the use of the ComputationGraph a, this version has skip connections between the first and output layers,
 * in order to show how this configuration is done. In practice, this means we have the following types of connections:
 * (a) first layer -> second layer connections
 * (b) first layer -> output layer connections
 * (c) second layer -> output layer connections
 *
 * @author Alex Black
 *
 * modified by kepricon on 17. 3. 28.
 *
 */
public class W2VSentiment implements TestableModel {

    private int vectorSize = 300;   //Size of the word vectors. 300 in the Google News model

    public W2VSentiment(){

    }

    public MultiLayerConfiguration conf() {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .updater(Updater.ADAM).adamMeanDecay(0.9).adamVarDecay(0.999)
                .regularization(true).l2(1e-5)
                .weightInit(WeightInit.XAVIER)
                .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue).gradientNormalizationThreshold(1.0)
                .learningRate(2e-2)
                .list()
                .layer(0, new GravesLSTM.Builder().nIn(vectorSize).nOut(256)
                        .activation(Activation.TANH).build())
                .layer(1, new RnnOutputLayer.Builder().activation(Activation.SOFTMAX)
                        .lossFunction(LossFunctions.LossFunction.MCXENT).nIn(256).nOut(2).build())
                .pretrain(false).backprop(true).build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
        net.setListeners(new PerformanceListener(10, true));

        return conf;
    }

    @Override
    public Model init() {
        MultiLayerNetwork network = new MultiLayerNetwork(conf());
        network.init();
        return network;
    }
}
