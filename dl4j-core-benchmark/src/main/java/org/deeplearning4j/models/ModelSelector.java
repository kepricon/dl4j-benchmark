package org.deeplearning4j.models;

import com.beust.jcommander.ParameterException;
import org.deeplearning4j.models.cnn.*;
import org.deeplearning4j.models.cnn.VGG16;
import org.deeplearning4j.models.rnn.W2VSentiment;

import java.util.HashMap;
import java.util.Map;

/**
 * Helper class for easily selecting multiple models for benchmarking.
 */
public class ModelSelector {
    public static Map<ModelType,TestableModel> select(ModelType modelType, int height, int width, int channels, int numLabels, int seed, int iterations) {
        Map<ModelType,TestableModel> netmap = new HashMap<>();

        switch(modelType) {
            case ALL:
                netmap.putAll(ModelSelector.select(ModelType.CNN, height, width, channels, numLabels, seed, iterations));
                netmap.putAll(ModelSelector.select(ModelType.RNN, height, width, channels, numLabels, seed, iterations));
                break;
            // CNN models
            case CNN:
                netmap.putAll(ModelSelector.select(ModelType.ALEXNET, height, width, channels, numLabels, seed, iterations));
                netmap.putAll(ModelSelector.select(ModelType.GOOGLELENET, height, width, channels, numLabels, seed, iterations));
                netmap.putAll(ModelSelector.select(ModelType.VGG16, height, width, channels, numLabels, seed, iterations));
                netmap.putAll(ModelSelector.select(ModelType.LENET, height, width, channels, numLabels, seed, iterations));

                break;
            case ALEXNET:
                netmap.put(ModelType.ALEXNET, new AlexNet(height, width, channels, numLabels, seed, iterations));
                break;
            case GOOGLELENET:
                netmap.put(ModelType.GOOGLELENET, new GoogleLeNet(height, width, channels, numLabels, seed, iterations));
                break;
            case INCEPTIONRESNETV1:
                netmap.put(ModelType.INCEPTIONRESNETV1, new InceptionResNetV1(height, width, channels, numLabels, seed, iterations));
            case FACENETNN4:
                netmap.put(ModelType.FACENETNN4, new FaceNetNN4(height, width, channels, numLabels, seed, iterations));
                break;
            case VGG16:
                netmap.put(ModelType.VGG16, new VGG16(height, width, channels, numLabels, seed, iterations));
                break;
            case LENET:
                netmap.put(ModelType.LENET, new LeNet(height, width, channels, numLabels, seed, iterations));
                break;
            // RNN models
            case RNN:
                netmap.put(ModelType.W2VSENTIMENT, new W2VSentiment());
                break;
            case W2VSENTIMENT:
                netmap.put(ModelType.W2VSENTIMENT, new W2VSentiment());
                break;

            default:
//                // do nothing
        }

        if(netmap.size()==0) throw new ParameterException("Zero models have been selected for benchmarking.");

        return netmap;
    }
}
