package org.deeplearning4j.models;

import org.apache.commons.lang.ArrayUtils;
import org.deeplearning4j.models.cnn.AlexNet;
import org.deeplearning4j.models.cnn.LeNet;
import org.deeplearning4j.models.cnn.VGG16;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;

/**
 * Created by justin on 3/21/17.
 */
public class ModelSelector {
    public static MultiLayerNetwork[] select(ModelType modelType, int height, int width, int channels, int numLabels, int seed, int iterations) {
        switch(modelType) {
            case ALL:
                return (MultiLayerNetwork[]) ArrayUtils.addAll(
                        ModelSelector.select(ModelType.CNN, height, width, channels, numLabels, seed, iterations),
                        ModelSelector.select(ModelType.RNN, height, width, channels, numLabels, seed, iterations));
            // CNN models
            case CNN:
                return new MultiLayerNetwork[]{
                        new AlexNet(height, width, channels, numLabels, seed, iterations).init(),
                        new LeNet(height, width, channels, numLabels, seed, iterations).init(),
                        new VGG16(height, width, channels, numLabels, seed, iterations).init()};
            // RNN models
            case RNN:
                return new MultiLayerNetwork[]{};
        }
    }
}
