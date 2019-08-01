import weka.classifiers.bayes.NaiveBayes;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class WekaVendas {

	public static void main(String[] args) throws Exception {
		// TODO Auto-generated method stub
		
		
		DataSource ds = new DataSource("C:\\mining\\vendas.arff");
		Instances ins = ds.getDataSet();
		//System.out.println(ins.toString());
		
		ins.setClassIndex(3);
		
		NaiveBayes nb = new NaiveBayes();
		nb.buildClassifier(ins);
		Instance novo = new DenseInstance(4);
		novo.setDataset(ins);
		
		novo.setValue(0, "M");
		novo.setValue(1, "20-39");
		novo.setValue(2, "Nao");
		
		double probabilidade[] = nb.distributionForInstance(novo);
		System.out.println("SIM: " +probabilidade[1]);
		System.out.println("NÃO: " +probabilidade[0]);
		

	}

}
