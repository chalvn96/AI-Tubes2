/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package tubesweka;

import java.io.Serializable;
import static java.lang.Math.exp;
import static java.lang.Math.pow;
import java.util.Random;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author taufic
 */
public class FFNN extends AbstractClassifier implements Classifier, Serializable {
    private double bias;
    private double learningRate;
    private double threshold;
    
    private int jumlahData;
    private int jumlahAtributAsli;
    private int jumlahKelas;
    private final int jumlahHL;
    private final int jumlahNeuron;
    
    private double[][] weightHL;
    private double[][] weightOp;
    
    public FFNN(int jumlahHL, int jumlahNeuron) {
        this.jumlahHL = jumlahHL;
        this.jumlahNeuron = jumlahNeuron;
    }

    private double sigmaHL (double[] weight, Instance ins) {
        double x = 0.0;
        for (int i = 0; i < jumlahAtributAsli; i++) {
            if (i != ins.classIndex()) {
                x += weight[i] * ins.value(i);
            }
        }
        x += weight[jumlahAtributAsli];
        return x;
    }
    
    private double sigmaOp (double[] weight, double[] hiddenlayer) {
        double x = 0.0;
        for (int i = 0; i < jumlahNeuron; i++) {
            x += weight[i] * hiddenlayer[i];
        }
        x += weight[jumlahNeuron];
        return x;
    }
    
    private double sigmoid (double x) {
        return 1 / (1 + exp(x));
    }
    
    private double errorOp (double output, double target) {
        return output * (1 - output) * (target - output);
    }
    
    private double errorHL (double hiddenlayer, double[] errorOp, double[][] weight, int index) {
        double x = 0.0;
        for (int i = 0; i < jumlahKelas; i++) {
            x += errorOp[i] * weight[i][index];
        }
        return hiddenlayer * (1 - hiddenlayer) * x;
    }
    
    private double sigmaError (double[] error) {
        double x = 0.0;
        for (int i = 0; i < jumlahKelas; i++) {
            x += error[i];
        }
        return x;
    }
    
    private double sigmaErrorSP (double[] error) {
        double x = 0.0;
        for (int i = 0; i < jumlahKelas; i++) {
            x += error[i];
        }
        return x;
    }
    
    private double updateWeight (double weight, double error, double input) {
        return weight + (learningRate * input * error);
    }
    
    private double errorSP (double output, double target) {
        return pow(target - output, 2.0) / 2.0;
    }
    
    @Override
    public void buildClassifier(Instances data) throws Exception {
        //data sudah numerik
        jumlahData = data.numInstances();
        jumlahAtributAsli = data.numAttributes();
        
        //tentukan jumlah neuron output
        jumlahKelas = data.attribute(data.classIndex()).numValues();
        
        //initialize bias and learning rate;
        Random r = new Random();
        bias = 1.0;
        learningRate = 0.9;
        threshold = 0.01;
        int limit = 1000;
        
        if (jumlahHL == 1) {
            //initialisasi random weight untuk atribut -> hidden layer
            weightHL = new double[jumlahNeuron][jumlahAtributAsli + 1];
            for (int i = 0; i < jumlahNeuron; i++) {
                for (int j = 0; j < jumlahAtributAsli + 1; j++) { //+1 untuk bias
                    if (j != data.classIndex()) {
                        weightHL[i][j] = (r.nextDouble() * 2) - 1;               
                    }
                }
            }
            
            //inisialisasi rndom weight untuk hidden layer -> output
            weightOp = new double[jumlahKelas][jumlahNeuron + 1];
            for (int i = 0; i < jumlahKelas; i++) {
                for (int j = 0; j < jumlahNeuron + 1; j++) { //+1 untuk bias
                    weightOp[i][j] = (r.nextDouble() * 2) - 1;
                }
            }
            
            //mulai FFNN
            double[] hiddenLayer = new double[jumlahNeuron];
            double[] output = new double[jumlahKelas];
            double totalError;
            int count = 0;
            do {
                count++;
                totalError = 0.0;
                //learning
                for (int nIns = 0; nIns < jumlahData; nIns++) {
                    Instance ins = data.instance(nIns);
                    
                    //hitung hiddenLayer
                    for (int i = 0; i < jumlahNeuron; i++) {
                        hiddenLayer[i] = sigmoid(sigmaHL(weightHL[i], ins));
                    }
                    
                    //hitung output layer
                    for (int i = 0; i < jumlahKelas; i++) {
                        output[i] = sigmoid(sigmaOp(weightOp[i], hiddenLayer));
                    }
                    
                    //hitung error masing-masing output
                    double[] errorOp = new double[jumlahKelas];
                    for (int i = 0; i < jumlahKelas; i++) {
                        errorOp[i] = errorOp(output[i], ins.value(ins.classIndex()));
                    }
                    
                    //hitung error masing-masing hidden layer
                    double[] errorHL = new double[jumlahNeuron];
                    for (int i = 0; i < jumlahNeuron; i++) {
                        errorHL[i] = errorHL(hiddenLayer[i], errorOp, weightOp, i);
                    }
                    
                    //update setiap weight output
                    for (int i = 0; i < jumlahKelas; i++) {
                        for (int j = 0; j < jumlahNeuron; j++) {
                            weightOp[i][j] = updateWeight(weightOp[i][j], errorOp[i], hiddenLayer[j]);
                        }
                        weightOp[i][jumlahNeuron] = updateWeight(weightOp[i][jumlahNeuron], errorOp[i], bias);
                    }
                    
                    //update setiap weight hidden layer
                    for (int i = 0; i < jumlahNeuron; i++) {
                        for (int j = 0; j < jumlahAtributAsli; j++) {
                            if (j != data.classIndex()) {
                                weightHL[i][j] = updateWeight(weightHL[i][j], errorHL[i], ins.value(j));                        
                            }
                        }
                        weightHL[i][jumlahAtributAsli] = updateWeight(weightHL[i][jumlahAtributAsli], errorHL[i], bias);
                    }
                }
                
                //testing
                for (int nIns = 0; nIns < jumlahData; nIns++) {
                    Instance ins = data.instance(nIns);
                    
                    //hitung hiddenLayer
                    for (int i = 0; i < jumlahNeuron; i++) {
                        hiddenLayer[i] = sigmoid(sigmaHL(weightHL[i], ins));
                    }
                    
                    //hitung output layer
                    for (int i = 0; i < jumlahKelas; i++) {
                        output[i] = sigmoid(sigmaOp(weightOp[i], hiddenLayer));
                    }
                    
                    //hitung error masing-masing output
                    double[] errorOp = new double[jumlahKelas];
                    for (int i = 0; i < jumlahKelas; i++) {
                        errorOp[i] = errorOp(output[i], ins.value(ins.classIndex()));
                    }
                    //2System.out.println("        sigmaError= "+sigmaError(errorOp));
                    totalError += sigmaError(errorOp);
                }
                System.out.println("totalError = " + totalError);
            } while ((totalError > threshold)&&(count<limit));
            
        } else { //jumlahHL == 0 (single perceptron)
            //initialisasi random weight untuk atribut
            weightOp = new double[jumlahKelas][jumlahAtributAsli + 1];
            for (int i = 0; i < jumlahKelas; i++) {
                for (int j = 0; j < jumlahAtributAsli + 1; j++) {
                    if (j != data.classIndex()) {
                        weightOp[i][j] = (r.nextDouble() * 2) - 1;
                    }
                }
            }
            
            //mulai FFNN
            double[] output = new double[jumlahKelas];
            double totalError;
            do {
                totalError = 0.0;
                //learning
                for (int nIns = 0; nIns < jumlahData; nIns++) {
                    Instance ins = data.instance(nIns);
                    
                    //hitung output layer
                    for (int i = 0; i < jumlahKelas; i++) {
                        output[i] = sigmoid(sigmaHL(weightOp[i], ins));
                    }
                    
                    //hitung error masing-masing output
                    double[] errorOp = new double[jumlahKelas];
                    for (int i = 0; i < jumlahKelas; i++) {
                        errorOp[i] = errorOp(output[i], ins.value(ins.classIndex()));
                    }
                    
                    //update setiap weight output
                    for (int i = 0; i < jumlahKelas; i++) {
                        for (int j = 0; j < jumlahAtributAsli; j++) {
                            if (j != data.classIndex()) {
                                weightOp[i][j] = updateWeight(weightOp[i][j], errorOp[i], ins.value(j));
                            }
                        }
                        weightOp[i][jumlahAtributAsli] = updateWeight(weightOp[i][jumlahAtributAsli], errorOp[i], bias);
                    }
                }
                
                //testing
                for (int nIns = 0; nIns < jumlahData; nIns++) {
                    Instance ins = data.instance(nIns);
                    
                    //hitung output layer
                    for (int i = 0; i < jumlahKelas; i++) {
                        output[i] = sigmoid(sigmaHL(weightOp[i], ins));
                    }
                    
                    //hitung error masing-masing output
                    double[] errorOp = new double[jumlahKelas];
                    for (int i = 0; i < jumlahKelas; i++) {
                        errorOp[i] = errorSP(output[i], ins.value(ins.classIndex()));
                    }
                    totalError += sigmaError(errorOp);
                } 
            } while (totalError > threshold);
        }
    }

    @Override
    public double classifyInstance(Instance ins) throws Exception {
        double[] output = distributionForInstance(ins);
        
        //cari max dari output
        double max = output[0];
        int index = 0;
        for (int i = 1; i < jumlahKelas; i++) {
            if (max < output[i]) {
                max = output[i];
                index = i;
            }
        }
        return (double) index;
    }

    @Override
    public double[] distributionForInstance(Instance ins) throws Exception {
        double[] hiddenLayer = new double[jumlahNeuron];
        double[] output = new double[jumlahKelas];
        
        //hitung hiddenLayer
        for (int i = 0; i < jumlahNeuron; i++) {
            hiddenLayer[i] = sigmoid(sigmaHL(weightHL[i], ins));
        }

        //hitung output layer
        for (int i = 0; i < jumlahKelas; i++) {
            output[i] = sigmoid(sigmaOp(weightOp[i], hiddenLayer));
        }
        
        return output;
    }

    @Override
    public Capabilities getCapabilities() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }
}
