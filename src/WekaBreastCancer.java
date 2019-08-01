import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.trees.HoeffdingTree;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class WekaBreastCancer {

	public static void main(String[] args) throws Exception {
		// TODO Auto-generated method stub
		
		DataSource ds = new DataSource("C:\\mining\\breast-cancer.arff");
		Instances ins = ds.getDataSet();
		
		//System.out.println(ins.toString());
		
		ins.setClassIndex(9);
		
		//NaiveBayes nb = new NaiveBayes();
		//nb.buildClassifier(ins);
		//Instance novo = new DenseInstance(9);
		//novo.setDataset(ins);

		//RandomForest nb = new RandomForest();
		
		//HoeffdingTree nb = new HoeffdingTree();
		
		//System.out.println(eval.toSummaryString());
		//System.out.println("-----------------------");
	//	System.out.println(eval.toClassDetailsString());
		//System.out.println("-----------------------");		
		//System.out.println(eval.toMatrixString());
	

		
		J48 nb = new J48();
		nb.getUnpruned();
		nb.buildClassifier(ins);
		Instance novo = new DenseInstance(9);
		novo.setDataset(ins);
		Evaluation eval = new Evaluation(ins);
		
		novo.setValue(0, "40-49"); //idade
		novo.setValue(1, "premeno"); //menopausa
		novo.setValue(2, "15-19"); //tamanho do tumor 
		novo.setValue(3, "0-2"); //inv-nodes
		novo.setValue(4, "yes"); //node-caps
		novo.setValue(5, "3"); //deg-malig
		novo.setValue(6, "right"); //mama
		novo.setValue(7, "left_up"); //breast-quad
		novo.setValue(8, "no"); //irradiação
		
		double probabilidade[] = nb.distributionForInstance(novo);
	
		System.out.println("Probabilidade de ter Câncer: " +probabilidade[1]);
		System.out.println("Probabilidade de não ter Câncer: " +probabilidade[0]);
		System.out.println(eval.kappa());
	    //Mostra oSystem.out.println(eval.toSummaryString());
			System.out.println("-----------------------");
			System.out.println(eval.toClassDetailsString());
			System.out.println("-----------------------");		
			System.out.println(eval.toMatrixString());

		

		
		

	}

}
