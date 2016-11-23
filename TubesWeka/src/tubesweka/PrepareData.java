/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package tubesweka;

import java.util.Scanner;
import java.util.logging.Level;
import java.util.logging.Logger;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;

/**
 *
 * @author taufic
 */
public class PrepareData {
    private Instances data;
    
    public PrepareData() {
        //cara mengambil data
        try {
            Scanner userInput = new Scanner(System.in);
            String inputData;
            System.out.print("masukan dataset yang ingin dibaca (.arff): ");
            inputData = userInput.nextLine();
            ConverterUtils.DataSource dataSource = new ConverterUtils.DataSource("testcase\\" + inputData + ".arff");
            data = dataSource.getDataSet();
//            int input;
//            System.out.print("Masukan class index dari user : ");
//            input = userInput.nextInt();
//            data.setClassIndex(input - 1);
        } catch (Exception ex) {
                Logger.getLogger(PrepareData.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
    public void printData() {
        System.out.println(data.toSummaryString());
    }
    
    public Instances getData() {
        return data;
    }
}
