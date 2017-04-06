package org.deeplearning4j.listeners;

import org.bytedeco.javacpp.cuda;
import org.bytedeco.javacpp.cudnn;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.factory.Nd4j;
import oshi.SystemInfo;
import oshi.hardware.HardwareAbstractionLayer;
import oshi.software.os.OperatingSystem;

import java.text.DecimalFormat;
import java.util.*;
import java.util.concurrent.TimeUnit;

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
    private String cudaVersion;
    private String cudnnVersion;
    private int numParams;
    private int numLayers;
    private long iterations;
    private long totalIterationTime;
    private double totalSamplesSec;
    private double totalBatchesSec;
    private double avgFeedForward;
    private double avgBackprop;
    private long avgUpdater;

    public BenchmarkReport(String name, String description) {
        this.name = name;
        this.description = description;

        Properties env = Nd4j.getExecutioner().getEnvironmentInformation();

        backend = env.get("backend").toString();
        cpuCores = env.get("cores").toString();
        blasVendor = env.get("blas.vendor").toString();

        if(backend.equals("CUDA")){
            cudaVersion = String.valueOf(cuda.__CUDA_API_VERSION);

            try {
                cudnnVersion = String.valueOf(cudnn.cudnnGetVersion());
            }catch (UnsatisfiedLinkError e){
                cudnnVersion = "n/a";
            }
        }

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

    public String getModelSummary() { return modelSummary; }

    public String toString() {
        DecimalFormat df = new DecimalFormat("#.##");

        SystemInfo sys = new SystemInfo();
        OperatingSystem os = sys.getOperatingSystem();
        HardwareAbstractionLayer hardware = sys.getHardware();

        ArrayList<String[]> table = new ArrayList<String[]>();
        table.add( new String[] { "Name", name } );
        table.add( new String[] { "Description", description } );
        table.add( new String[] { "Operating System",
                os.getManufacturer()+" "+
                os.getFamily()+" "+
                os.getVersion().getVersion() } );
        table.add( new String[] { "Devices", devices().get(0) } );
        table.add( new String[] { "CPU Cores", cpuCores } ) ;
        table.add( new String[] { "Backend", backend } );
        table.add( new String[] { "BLAS Vendor", blasVendor } );
        if(backend.equals("CUDA")){
            table.add( new String[] { "CUDA Version", cudaVersion } );
            table.add( new String[] { "CUDNN Version", cudnnVersion } );
        }
        table.add( new String[] { "Total Params", Integer.toString(numParams) } );
        table.add( new String[] { "Total Layers", Integer.toString(numLayers) } );
        table.add( new String[] { "Avg Feedforward (ms)", df.format(avgFeedForward) } );
        table.add( new String[] { "Avg Backprop (ms)", df.format(avgBackprop) } );
        table.add( new String[] { "Avg Iteration (ms)", df.format(avgIterationTime()) } );
        table.add( new String[] { "Avg Samples/sec", df.format(avgSamplesSec()) } );
        table.add( new String[] { "Avg Batches/sec", df.format(avgBatchesSec()) } );

        StringBuilder sb = new StringBuilder();

        for (final Object[] row : table) {
            sb.append(String.format("%28s %45s\n", row));
        }

        return sb.toString();
    }

}
