package org.deeplearning4j.listeners;

import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.factory.Nd4j;
import oshi.SystemInfo;
import oshi.hardware.HardwareAbstractionLayer;
import oshi.software.os.OperatingSystem;

import java.text.DecimalFormat;
import java.util.*;

/**
 * Reporting for BenchmarkListener.
 *
 * @author Justin Long (@crockpotveggies)
 */
public class BenchmarkReport {

    private String name;
    private String description;
    private List<String> devices = new ArrayList<>();
    private String backend;
    private String cpuCores;
    private String blasVendor;
    private String modelSummary;
    private int numParams;
    private int numLayers;
    private long iterations;
    private long totalIterationTime;
    private double totalSamplesSec;
    private double totalBatchesSec;
    private double avgFeedForward;
    private double avgBackprop;
    private long avgUpdater;
    private long totalTrainingTime;

    public BenchmarkReport(String name, String description) {
        this.name = name;
        this.description = description;

        Properties env = Nd4j.getExecutioner().getEnvironmentInformation();

        backend = env.get("backend").toString();
        cpuCores = env.get("cores").toString();
        blasVendor = env.get("blas.vendor").toString();

        // if CUDA is present, add GPU information
        try {
            List devicesList = (List) env.get("cuda.devicesInformation");
            Iterator deviceIter = devicesList.iterator();
            while (deviceIter.hasNext()) {
                Map dev = (Map) deviceIter.next();
                devices.add(dev.get("cuda.deviceName") + " " + dev.get("cuda.deviceMajor") + " " + dev.get("cuda.deviceMinor") + " " + dev.get("cuda.totalMemory"));
            }
        } catch(Exception e) {
            SystemInfo sys = new SystemInfo();
            devices.add(sys.getHardware().getProcessor().getName());
        }
    }

    public void setModel(Model model) {
        this.numParams = model.numParams();

        if(model instanceof MultiLayerNetwork) {
            this.modelSummary = ((MultiLayerNetwork) model).summary();
            this.numLayers = ((MultiLayerNetwork) model).getnLayers();
        }
        if(model instanceof ComputationGraph) {
            this.modelSummary = ((ComputationGraph) model).summary();
            this.numLayers = ((ComputationGraph) model).getNumLayers();
        }
    }

    public void setIterations(long iterations) { this.iterations = iterations; }

    public void addIterationTime(long iterationTime) { totalIterationTime += iterationTime; }

    public void addSamplesSec(double samplesSec) { totalSamplesSec += samplesSec; }

    public void addBatchesSec(double batchesSec) { totalBatchesSec += batchesSec; }

    public void setAvgFeedForward(double feedForwardTime) { avgFeedForward = feedForwardTime; }

    public void setAvgBackprop(double backpropTime) { this.avgBackprop = backpropTime; }

    public void setAvgUpdater(long updaterTime) { this.avgUpdater = updaterTime; }

    public List<String> devices() { return devices; }

    public double avgIterationTime() { return (double) totalIterationTime / (double) iterations; }

    public double avgSamplesSec() { return totalSamplesSec / (double) iterations; }

    public double avgBatchesSec() { return totalBatchesSec / (double) iterations; }

    public double avgFeedForward() { return avgFeedForward; }

    public double avgBackprop() { return avgBackprop; }

    public void setTotalTrainingTime(long totalTrainingTime) {
        this.totalTrainingTime = totalTrainingTime;
    }

    public long getTotalIterationTime() { return this.totalTrainingTime; }

    public String getModelSummary() { return modelSummary; }

    public String toString() {
        DecimalFormat df = new DecimalFormat("#.##");

        SystemInfo sys = new SystemInfo();
        OperatingSystem os = sys.getOperatingSystem();
        HardwareAbstractionLayer hardware = sys.getHardware();

        final Object[][] table = new String[15][];
        table[0] = new String[] { "Name", name };
        table[1] = new String[] { "Description", description };
        table[2] = new String[] { "Operating System",
                os.getManufacturer()+" "+
                os.getFamily()+" "+
                os.getVersion().getVersion() };
        table[3] = new String[] { "Devices", devices().get(0) };
        table[4] = new String[] { "CPU Cores", cpuCores };
        table[5] = new String[] { "Backend", backend };
        table[6] = new String[] { "BLAS Vendor", blasVendor };
        table[7] = new String[] { "Total Params", Integer.toString(numParams) };
        table[8] = new String[] { "Total Layers", Integer.toString(numLayers) };
        table[9] = new String[] { "Total Training Time (ms)", df.format(totalTrainingTime) };
        table[10] = new String[] { "Avg Feedforward (ms)", df.format(avgFeedForward) };
        table[11] = new String[] { "Avg Backprop (ms)", df.format(avgBackprop) };
        table[12] = new String[] { "Avg Iteration (ms)", df.format(avgIterationTime()) };
        table[13] = new String[] { "Avg Samples/sec", df.format(avgSamplesSec()) };
        table[14] = new String[] { "Avg Batches/sec", df.format(avgBatchesSec()) };

        StringBuilder sb = new StringBuilder();

        for (final Object[] row : table) {
            sb.append(String.format("%28s %45s\n", row));
        }

        return sb.toString();
    }

}
