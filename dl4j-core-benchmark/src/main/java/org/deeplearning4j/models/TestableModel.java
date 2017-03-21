package org.deeplearning4j.models;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;

/**
 * Generic interface for testable models.
 */
public interface TestableModel {

    public MultiLayerConfiguration conf();

    public MultiLayerNetwork init();
}
