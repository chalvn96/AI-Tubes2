/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package tubesweka;

import java.io.Serializable;
import static java.lang.Double.isNaN;
import java.util.HashMap;
import java.util.Map;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author lenovo
 */
//class Kelas implements Serializable {
//    private String element0;
//    private Integer element1;
//    
//    public Kelas(){
//        element0 = new String();
//        element1 = null;
//    }
//    
//}

class Trice<K, L, V> implements Serializable {
    private final K element0;
    private final L element1;
    private final V element2;

    public Trice(){
        this.element0 = null;
        this.element1 = null;
        this.element2 = null;
    }
    
    public <K, L, V> Trice<K, L, V> createTrice(K element0, L element1, V element2) {
        return new Trice<>(element0, element1, element2);
    }

    public Trice(K element0, L element1, V element2) {
        this.element0 = element0;
        this.element1 = element1;
        this.element2 = element2;
    }

    public K getElement0() {
        return element0;
    }

    public L getElement1() {
        return element1;
    }
    
    public V getElement2() {
        return element2;
    }
    @Override
    public boolean equals(Object o)
    {
        if (o instanceof Trice) {
           Trice<?, ?, ?> trice = (Trice<?, ?, ?>) o;
           return this.element0.equals(trice.element0) && this.element1.
                   equals(trice.element1) && this.element2.equals(trice.element2);
        }
        return false;
    }
    
    @Override
    public int hashCode(){
        return (int) element0.hashCode() * element1.hashCode() 
                * element2.hashCode();
    }

}


public class NaiveBayes extends AbstractClassifier implements Classifier, Serializable {
    private Map< Trice<String,String,String>, Integer >  m; //data
    private Map<String, Integer> mKelas; //hitung jumlah per kelas
    private Map<String, Double> mHitung; //hitung peluang
    private int dataSize;
    
    public NaiveBayes() {
        
    }
    @Override
    public void buildClassifier(Instances data) throws Exception {
        //generate
        m = new HashMap(); //data
        mKelas = new HashMap(); //hitung jumlah per kelas
        mHitung = new HashMap(); //hitung peluang
       
        int jumlahAtributAsli = data.numAttributes();
        int indexKelas = data.classIndex();
        //int jumlahAtribut = jumlahAtributAsli - 1;
        System.out.println(jumlahAtributAsli - 1);
        System.out.println(indexKelas);
//        paling ujung brrt jumlahAtribut-1
        int jumlahValuePerAtribut;
//        int jumlahKelas = data.attribute(jumlahAtribut).numValues();
        int jumlahKelas = data.attribute(indexKelas).numValues();
        System.out.println("tes1");
        dataSize = data.size();
//        for (int i = 0; i < jumlahAtribut; i++){//karena yg paling ujung itu kelas
        for (int i = 0; i < jumlahAtributAsli; i++){//karena yg paling ujung itu kelas
            if (i == indexKelas) {
                
            } else {
                jumlahValuePerAtribut =  data.attribute(i).numValues();
                for (int j = 0; j < jumlahValuePerAtribut; j++) {
                    for (int k = 0; k < jumlahKelas; k++){

                        Trice<String, String, String> trice = new Trice();
//                        pair = pair.createPair(data.attribute(i).value(j), data.attribute(jumlahAtribut).value(k));
                        trice = trice.createTrice(data.attribute(i).name(), data.attribute(i).value(j), data.attribute(indexKelas).value(k));
                        
                        m.put(trice,0);
                    }
                }
            }
        }
        System.out.println("tes3");
       
        //init for kelas
        for (int i = 0; i < jumlahKelas; i++){
//            mKelas.put(data.attribute(jumlahAtribut).value(i), 0);
            mKelas.put(data.attribute(indexKelas).value(i), 0);
//            System.out.println(data.attribute(indexKelas).value(i));
//            mHitung.put(data.attribute(jumlahAtribut).value(i),0.0);
            mHitung.put(data.attribute(indexKelas).value(i),0.0);
        }
        System.out.println("tes4");
        //update isi atribut
        for (int i = 0; i < data.size(); i++) {
            String hKelas = new String();
            hKelas = data.instance(i).stringValue(indexKelas);
//            hKelas = data.instance(i).stringValue(jumlahAtribut);
//            System.out.println(data.instance(i).stringValue(indexKelas));
//            System.out.println(data.instance(i).stringValue(jumlahAtribut));
            if (mKelas.containsKey(hKelas)) {
                mKelas.put(hKelas, mKelas.get(hKelas) + 1);
//                System.out.println("hKelas " + hKelas);
//                System.out.println("mkelas " + mKelas.get(hKelas) + 1);
            }
            for (int j = 0; j < jumlahAtributAsli; j++){//dikurang 2 karena mulai dari 0 dan atribut paling ujung itu kelas
                if (j != indexKelas) {
                    Trice<String, String, String> x = new Trice();
//                   x = x.createPair(data.instance(i).stringValue(j), data.instance(i).stringValue(jumlahAtribut));
                    x = x.createTrice(data.attribute(j).name(), data.instance(i).stringValue(j), data.instance(i).stringValue(indexKelas));
                    if(m.containsKey(x)) {
                        m.put(x, m.get(x) + 1);
                    }
                    else {
                       System.out.println("Not Contained");
                    }
                }
            }
        }
       
       //print
        System.out.println("===========");
        
        for (Map.Entry<Trice<String,String,String>,Integer> entry : m.entrySet()) {
            Trice<String,String,String> key = entry.getKey();
            Integer count = entry.getValue();
            // do stuff
            System.out.println(key.getElement0() +" "+ key.getElement1() +" " + key.getElement2() +" "+count);
            System.out.println(m.get(key));
        }
        System.out.println("============");
       
        for (Map.Entry<String,Integer> entry : mKelas.entrySet()) {
            String key = entry.getKey();
            Integer count = entry.getValue();
            // do stuff
            System.out.println(key+" "+count);
        }
        
        //mengolah fread (yg mau diclassify
//        for (int i = 0; i < jumlahAtributAsli; i++){//karna yg paling ujung iu kelas nah ini masalah krn ga selalu yg paling ujung itu kelas
//           //String fKelas = new String();
//           //fKelas = data.instance(i).stringValue(jumlahAtribut);
//           if (i == indexKelas) {
//                Double hitung = 0.0;
//           }
//        }
    }

    @Override
    public double classifyInstance(Instance instnc) throws Exception {
        int jumlahAtributAsli = instnc.numAttributes();
        
        int jumlahData = 0;
        String kelasNow = "";
        double tempHitung;
        double hitung;
        double[] listclass = new double[instnc.attribute(instnc.classIndex()).numValues()];
        int max = 0;
        int nilaiMax = 0;
        for(int k = 0; k < instnc.attribute(instnc.classIndex()).numValues(); k++){
            hitung = 1.0;
            for(int i = 0; i < jumlahAtributAsli; i++) {
                if (i != instnc.classIndex()) {
                    Trice<String, String, String> cek = new Trice();
                    System.out.println("string value " + instnc.stringValue(i));
                    kelasNow = instnc.attribute(instnc.classIndex()).value(k);
                    cek = cek.createTrice(instnc.attribute(i).name(), instnc.stringValue(i), kelasNow);
                    System.out.println("cek " + m.get(cek));
                    System.out.println("kelas " + mKelas.get(kelasNow));
                    tempHitung = ((double) m.get(cek) / (double) mKelas.get(kelasNow));
                    System.out.println("tempHitung " + tempHitung);
                    hitung = hitung * tempHitung;
                    System.out.println("hitung " + hitung);
                }
            }
            hitung = hitung * ((double) mKelas.get(kelasNow) / (double) dataSize);
            System.out.println("hitung kedua " + hitung);
            listclass[k] = hitung;
            System.out.println(k + " " + listclass[k]);
            if(nilaiMax < hitung){
                max = k;
            }
        }
        
        return (double)max;
    }

    @Override
    public double[] distributionForInstance(Instance instnc) throws Exception {
        int jumlahAtributAsli = instnc.numAttributes();        
        String kelasNow = "";
        double tempHitung;
        double hitung;
        double[] listclass = new double[instnc.attribute(instnc.classIndex()).numValues()];
        int nilaiMax = 0;
        for(int k = 0; k < instnc.attribute(instnc.classIndex()).numValues(); k++){
            hitung = 1.0;
            for(int i = 0; i < jumlahAtributAsli; i++) {
                if (i != instnc.classIndex()) {
                    Trice<String, String, String> cek = new Trice();
                    //System.out.println("string value " + instnc.stringValue(i));
                    kelasNow = instnc.attribute(instnc.classIndex()).value(k);
                    cek = cek.createTrice(instnc.attribute(i).name(), instnc.stringValue(i), kelasNow);
                    //System.out.println("cek " + m.get(cek));
                    //System.out.println("kelas " + mKelas.get(kelasNow));
                    tempHitung = ((double) m.get(cek) / (double) mKelas.get(kelasNow));
                    //System.out.println("tempHitung " + tempHitung);
                    hitung = hitung * tempHitung;
                    //System.out.println("hitung " + hitung);
                }
            }
            hitung = hitung * ((double) mKelas.get(kelasNow) / (double) dataSize);
            
            if (isNaN(hitung)) {
                hitung = 0.0;
            }
            System.out.println("hitung kedua " + hitung);
            
            listclass[k] = hitung;
        }
        
        return listclass;
    }

    @Override
    public Capabilities getCapabilities() {
        
        Capabilities result = super.getCapabilities();
        result.disableAll();

        // attributes
        result.enable(Capabilities.Capability.NOMINAL_ATTRIBUTES);
        result.enable(Capabilities.Capability.NUMERIC_ATTRIBUTES);
        result.enable(Capabilities.Capability.MISSING_VALUES );

        // class
        result.enable(Capabilities.Capability.NOMINAL_CLASS);
        result.enable(Capabilities.Capability.MISSING_CLASS_VALUES);

        // instances
        result.setMinimumNumberInstances(0);

        return result;
    }
}
