package org.deeplearning4j.models;

import com.beust.jcommander.ParameterException;
import javafx.util.Pair;
import org.apache.commons.lang.ArrayUtils;
import org.deeplearning4j.models.cnn.AlexNet;
import org.deeplearning4j.models.cnn.LeNet;
import org.deeplearning4j.models.cnn.VGG16;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;

import java.util.HashMap;
import java.util.Map;

/**
 * Helper class for easily selecting multiple models for benchmarking.
 */
public class ModelSelector {
    public static Map<ModelType,MultiLayerNetwork> select(ModelType modelType, int height, int width, int channels, int numLabels, int seed, int iterations) {
        Map<ModelType,MultiLayerNetwork> netmap = new HashMap<>();

        switch(modelType) {
            case ALL:
                netmap.putAll(ModelSelector.select(ModelType.CNN, height, width, channels, numLabels, seed, iterations));
                netmap.putAll(ModelSelector.select(ModelType.RNN, height, width, channels, numLabels, seed, iterations));
            // CNN models
            case CNN:
                netmap.putAll(ModelSelector.select(ModelType.ALEXNET, height, width, channels, numLabels, seed, iterations));
                netmap.putAll(ModelSelector.select(ModelType.LENET, height, width, channels, numLabels, seed, iterations));
                netmap.putAll(ModelSelector.select(ModelType.VGG16, height, width, channels, numLabels, seed, iterations));
                return netmap;
            case ALEXNET:
                netmap.put(ModelType.ALEXNET, new AlexNet(height, width, channels, numLabels, seed, iterations).init());
            case LENET:
                netmap.put(ModelType.LENET, new LeNet(height, width, channels, numLabels, seed, iterations).init());
            case VGG16:
                netmap.put(ModelType.VGG16, new VGG16(height, width, channels, numLabels, seed, iterations).init());
            // RNN models
            case RNN:
//                // not yet
            default:
//                // do nothing
        }

        if(netmap.size()==0) throw new ParameterException("Zero models have been selected for benchmarking.");

        return netmap;
    }
}
