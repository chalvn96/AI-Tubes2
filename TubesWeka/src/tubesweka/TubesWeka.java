/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package tubesweka;

import java.util.Scanner;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Debug.Random;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.filters.Filter;
import weka.filters.supervised.attribute.Discretize;
import weka.filters.supervised.attribute.NominalToBinary;
import weka.filters.unsupervised.attribute.Normalize;
import weka.filters.unsupervised.instance.Randomize;

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
        //System.out.println(data);
        Evaluation eval = null;
<<<<<<< HEAD
//        Randomize random = new Randomize();
//        random.setInputFormat(data);
//        Instances randomize = Filter.useFilter(data, random);
=======
        Randomize random = new Randomize();
        random.setInputFormat(data);
        Instances randomize = Filter.useFilter(data, random);
//<<<<<<< HEAD
//        
//        NominalToBinary filter = new NominalToBinary();
//        //RenameNominalValues filter = new RenameNominalValues();
//        filter.setInputFormat(randomize);
//        Instances filterNominal = Filter.useFilter(randomize, filter);
//        //System.out.println(filterNominal);
//        Normalize filter1 = new Normalize();
//        filter1.setInputFormat(filterNominal);
//        Instances filteredData = Filter.useFilter(filterNominal, filter1);
//=======
//>>>>>>> 6c0f51c847581faf8a6a82bf560cbc492ce3ce7e
>>>>>>> 25731c79b0f3d44973f3fb171767acee6d8c808d
        //System.out.println(filteredData);
        if (n == 1) {//memilih model pembelajaran
            Instances datatest = data;
            System.out.println("Membaca datatesting...");
            System.out.print("Masukan model yang ingin digunakan :");
            inputUser = new Scanner(System.in);
            string = inputUser.nextLine();
            
            //Classification
            double result = 0.0;
//            Discretize filter = new Discretize();
//            filter.setInputFormat(datatest);
//            Instances filterDataSet = Filter.useFilter(datatest, filter);
//            
                Normalize normal = new Normalize();  
                normal.setInputFormat(data);
                Instances normalize = Filter.useFilter(data, normal);
                NominalToBinary ntb = new NominalToBinary();
                ntb.setInputFormat(normalize);
                Instances filterDataSet = Filter.useFilter(normalize, ntb);
        
            try { 
                Classifier classifier = (Classifier) SerializationHelper.read(string + ".model");
                //classifier.buildClassifier(filterDataSet);
                eval = new Evaluation(filterDataSet);
                
                eval.evaluateModel(classifier, filterDataSet);
                System.out.println(eval.toSummaryString("\nResults\n\n", false));                           
                System.out.println(eval.toClassDetailsString());                             
                System.out.println(eval.toMatrixString());
    //                int folds = 14;
//                eval.crossValidateModel(classifier, filteredData, folds,
//                        new Random(1));
//                for (int i = 0;  i < filterDataSet.numInstances(); i++) {
//                    result = classifier.classifyInstance(filterDataSet.instance(i));
//                    filterDataSet.instance(i).setClassValue(result);
//                    System.out.println(filterDataSet.classAttribute().value((int) result));
//                }
            } catch(Exception e) {
                e.printStackTrace();
            }
            
            
        } else if (n == 2) {//memilih dataset yang ingin dipelajari
            
            System.out.println("Pilih salah satu teknik pembelajaran :");
            System.out.println("1. Naive Bayes");
            System.out.println("2. FFNN");
            System.out.print("Masukan input: ");
            n = inputUser.nextInt();
            Classifier classifier = null;

            Normalize normal = new Normalize();
            normal.setInputFormat(data);
            Instances normalize = Filter.useFilter(data, normal);
            if (n == 1) {//NAIVE BAYES
                Discretize filter = new Discretize();
                filter.setInputFormat(normalize);
                Instances filteredData = Filter.useFilter(normalize, filter);
                
                classifier = new NaiveBayes();                
                eval = new Evaluation(filteredData);            
                
                System.out.println("Build Classifier");
                classifier.buildClassifier(filteredData);
                System.out.println("Evaluate model");
                int folds = 10;
//                eval.crossValidateModel(classifier, filteredData, folds,
//                        new Random(1));
                eval.evaluateModel(classifier, filteredData);
            } else if (n == 2) {//FFNN
                NominalToBinary ntb = new NominalToBinary();
                ntb.setInputFormat(normalize);
                Instances filteredData = Filter.useFilter(normalize, ntb);
                
                eval = new Evaluation(filteredData);            
                
                
                System.out.print("Masukkan jumlah hidden layer [0-1]: ");
                int jumlahHL = inputUser.nextInt(); //jumlah hidden layer
                int jumlahNeuron = 0;
                if (jumlahHL == 1) {
                    System.out.print("Masukkan jumlah neuron pada hidden layer: ");
                    jumlahNeuron = inputUser.nextInt(); //jumlah neuron pada hidden layer
                }
                classifier = new FFNN(jumlahHL, jumlahNeuron);
                classifier.buildClassifier(filteredData);
                System.out.println("SELESAI BUILD CLASSIFIER");
                int folds = 10; //10 folds
//                eval.crossValidateModel(classifier, filteredData, folds,
//                        new Random(1));
                eval.evaluateModel(classifier, filteredData);
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