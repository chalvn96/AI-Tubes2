/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package tubesweka;

import java.util.ArrayList;
import java.util.Scanner;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Attribute;
import weka.core.Debug.Random;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.filters.Filter;
import weka.filters.supervised.attribute.Discretize;
import weka.filters.unsupervised.attribute.Normalize;

/**
 *
 * @author taufic
 */
public class TubesWeka {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) throws Exception {
        
        //tampilan menu
        System.out.println("*************************************");
        System.out.println("***********Welcome to Weka***********");
        System.out.println("*************************************");
        System.out.println("Pilih salah satu menu dibawah ini :");
        System.out.println("1. Masukan model pembelajaran");
        System.out.println("2. Masukan dataset yang ingin dipelajari");
        System.out.println("3. Exit");
        System.out.print("Masukan input user : ");
        
        Scanner inputUser = new Scanner(System.in);
        String string = null;
        //input masukan menu no berapa
        int n = inputUser.nextInt();
        
        //Preparing data
        PrepareData preparedata = new PrepareData();
        Instances data = preparedata.getData();

        
        //Melakukan filter
        //Discretize filter = new Discretize();
        Normalize filter = new Normalize();  
        filter.setInputFormat(data);
        Instances filteredData = Filter.useFilter(data, filter);
        //System.out.println(filteredData);
//        Instances filteredData = data;
        if (n == 1) {
            Instances datatest = data;
            System.out.println("Membaca datatesting...");
            System.out.print("Masukan model yang ingin digunakan :");
            inputUser = new Scanner(System.in);
            string = inputUser.nextLine();
            //Classification
            double result = 0.0;
            Instances filterDataSet = Filter.useFilter(datatest, filter);
            
            try { 
                Classifier classifier = (Classifier) SerializationHelper.read(string);
                classifier.buildClassifier(filterDataSet);
                Evaluation eval = new Evaluation(filterDataSet);
                int folds = 14;
                eval.crossValidateModel(classifier, filteredData, folds,
                        new Random(1));
            } catch(Exception e) {
                e.printStackTrace();
            }
            System.out.println("Classification : ");
            System.out.println(filterDataSet.classAttribute().value((int) result));
            
        } else if (n == 2) {
            Evaluation eval = new Evaluation(filteredData);
            System.out.println("Pilih salah satu teknik pembelajaran :");
            System.out.println("1. Naive Bayes");
            System.out.println("2. FFNN");
            System.out.print("Masukan input: ");
            n = inputUser.nextInt();
            Classifier classifier = null;
            if (n == 1) {
                classifier = new NaiveBayes();
                
                System.out.println("Build Classifier");
                classifier.buildClassifier(filteredData);
                System.out.println("Evaluate model");
                int folds = 500;
                eval.crossValidateModel(classifier, filteredData, folds,
                        new Random(1));
                //eval.evaluateModel(classifier, filteredData);
            } else if (n == 2) {
                System.out.print("Masukkan jumlah hidden layer [0-1]: ");
                int jumlahHL = inputUser.nextInt(); //jumlah hidden layer
                int jumlahNeuron = 0;
                if (jumlahHL == 1) {
                    System.out.print("Masukkan jumlah neuron pada hidden layer: ");
                    jumlahNeuron = inputUser.nextInt(); //jumlah neuron pada hidden layer
                }
                classifier = new FFNN(jumlahHL, jumlahNeuron);
                classifier.buildClassifier(filteredData);
                int folds = 10;
                eval.crossValidateModel(classifier, filteredData, folds,
                        new Random(1));
//                eval.evaluateModel(classifier, filteredData);
            }
            
            //mengeluarkan hasil evaluasi
            System.out.println(eval.toSummaryString("\nResults\n\n", false));                           
            System.out.println(eval.toClassDetailsString());                             
            System.out.println(eval.toMatrixString());
            //menanyakan apakah hasil model pembelajaran perlu
            //disimpan atau tidak
            System.out.print("Pemodelan pembelajaran mau di save kah?(Y/N) : ");
            inputUser = new Scanner(System.in);
            string = inputUser.nextLine();
            if ("y".equals(string) || "Y".equals(string)) {
                System.out.print("Masukan nama file yang ingin disave : ");
                inputUser = new Scanner(System.in);
                string = inputUser.nextLine();
                weka.core.SerializationHelper.write(string + ".model",
                        classifier);
                System.out.println("file sudah disave");
            } else {
                System.out.println("file tidak disave");
            }
        
        }
    }
          
}