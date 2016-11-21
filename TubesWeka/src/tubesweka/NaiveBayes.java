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

//Trice menyimpan: nama atribut, value tertenu dari atribut yang hendak diolah,dan value dari kelas
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
    
    //getter
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
    {//mengembalikan true jika object yang dibandingkan dianggap sama
        if (o instanceof Trice) {
           Trice<?, ?, ?> trice = (Trice<?, ?, ?>) o;
           return this.element0.equals(trice.element0) && this.element1.
                   equals(trice.element1) && this.element2.equals(trice.element2);
        }
        return false;
    }
    
    @Override
    public int hashCode(){//mengembalikan hash Code
        return (int) element0.hashCode() * element1.hashCode() * element2.hashCode();
    }
}


public class NaiveBayes extends AbstractClassifier implements Classifier, Serializable {
    //melakukan pengerjaan NaiveBayes dengan implementasi sendiri
    
    private Map< Trice<String,String,String>, Integer >  m; //menyimpan data(sebagaimana dituliskan di atas) dan jumlah yang ditemui
    private Map<String, Integer> mKelas; //menyimpan hitung jumlah per kelas
    private Map<String, Double> mHitung; //menyimpan hitung peluang
    private int dataSize;
    
    public NaiveBayes() {
        
    }
    
    @Override
    public void buildClassifier(Instances data) throws Exception {//melakukan override untuk buildCassifier
        
        //generate (temp)
        m = new HashMap(); //data
        mKelas = new HashMap(); //hitung jumlah per kelas
        mHitung = new HashMap(); //hitung peluang
       
        int jumlahAtributAsli = data.numAttributes();
        int indexKelas = data.classIndex();
        int jumlahValuePerAtribut;
        int jumlahKelas = data.attribute(indexKelas).numValues();
        dataSize = data.size();
        
        /*System.out.println("Jumlah Atribut pembacaan: "+jumlahAtributAsli - 1);
        System.out.println("Kelas berada di index: "+indexKelas);*/
        
        for (int i = 0; i < jumlahAtributAsli; i++){//karena index start dr 0
            if (i == indexKelas) {
                //tidak perlu digenerate
            } else {
                jumlahValuePerAtribut =  data.attribute(i).numValues();
                for (int j = 0; j < jumlahValuePerAtribut; j++) {
                    for (int k = 0; k < jumlahKelas; k++){
                        Trice<String, String, String> trice = new Trice();
                        trice = trice.createTrice(data.attribute(i).name(), data.attribute(i).value(j), data.attribute(indexKelas).value(k));
                        m.put(trice,0);
                    }
                }
            }
        }
        //System.out.println("selesai generate data store");
       
        //init for kelas
        for (int i = 0; i < jumlahKelas; i++){
            mKelas.put(data.attribute(indexKelas).value(i), 0);
            mHitung.put(data.attribute(indexKelas).value(i),0.0);
        }
        //System.out.println("selesai generate class store");
        
        //update isi atribut sesuai pembacaan
        for (int i = 0; i < data.size(); i++) {
            String hKelas = new String();
            hKelas = data.instance(i).stringValue(indexKelas);
            if (mKelas.containsKey(hKelas)) {
                mKelas.put(hKelas, mKelas.get(hKelas) + 1);
            }
            for (int j = 0; j < jumlahAtributAsli; j++){
                if (j != indexKelas) {
                    Trice<String, String, String> x = new Trice();
                    x = x.createTrice(data.attribute(j).name(), data.instance(i).stringValue(j),
                            data.instance(i).stringValue(indexKelas));
                    if(m.containsKey(x)) {//jika data yang dibaca ada yang sesuai dengan hasil generate
                        m.put(x, m.get(x) + 1);
                    }
                    else {
                       System.out.println("Not Contained");
                    }
                }
            }
        }
       
        System.out.println("================");
        
        //Print
        /*
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
        }*/
    }

    @Override
    public double classifyInstance(Instance instnc) throws Exception {
        //melakukan override untuk pemodelan
        int jumlahAtributAsli = instnc.numAttributes();
        int jumlahData = 0;
        String kelasNow = "";
        double tempHitung;
        double hitung;
        double[] listclass = new double[instnc.attribute(instnc.classIndex()).numValues()]; //menyimpan daftar nilai
        int max = 0;
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
            //System.out.println("hitung kedua " + hitung);
            listclass[k] = hitung;
            //System.out.println(k + " " + listclass[k]);
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
                    tempHitung = ((double) m.get(cek) / (double) mKelas.get(kelasNow));
                    hitung = hitung * tempHitung; //hasil perkalian dari peluang atribut | value kelas
                }
            }
            hitung = hitung * ((double) mKelas.get(kelasNow) / (double) dataSize); //dikali peluang terwujudnya value kelas
            
            //penanganan bilangan NaN
            if (isNaN(hitung)) {
                 hitung = 0.0;
            }
            
            //assign nilai hitung ke list
            listclass[k] = hitung;
        }
        return listclass;
    }

    @Override
    public Capabilities getCapabilities() {
        //melakukan override getCapabilities
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
