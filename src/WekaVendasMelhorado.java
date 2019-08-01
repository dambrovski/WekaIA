import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.trees.RandomForest;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class WekaVendasMelhorado {

	public static void main(String[] args) throws Exception {
		// TODO Auto-generated method stub
		
		DataSource ds = new DataSource("C:\\mining\\vendas2.arff");
		Instances ins = ds.getDataSet();
		//System.out.println(ins.toString());
		
		ins.setClassIndex(3);
		
		RandomForest nb = new RandomForest();
		nb.buildClassifier(ins);
		Instance novo = new DenseInstance(4);
		novo.setDataset(ins);
		
		novo.setValue(0, "F");
		novo.setValue(1, "20-39");
		novo.setValue(2, "Sim");
		
		double probabilidade[] = nb.distributionForInstance(novo);
		System.out.println("SIM - vai gastar muito: " +probabilidade[1]);
		System.out.println("NÃO - nao vai gastar muito: " +probabilidade[0]);
		

	}

}
