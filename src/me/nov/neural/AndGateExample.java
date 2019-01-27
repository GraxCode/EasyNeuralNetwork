package me.nov.neural;

public class AndGateExample {
	public static void main(String[] args) {
		NeuralNetwork nn = new NeuralNetwork();
		System.out.println("Random weights\n--------");
		System.out.println("1 0 -> " + nn.compute(new double[] { 1, 0 }));
		System.out.println("0 0 -> " + nn.compute(new double[] { 0, 0 }));
		System.out.println("0 1 -> " + nn.compute(new double[] { 0, 1 }));
		System.out.println("1 1 -> " + nn.compute(new double[] { 1, 1 }));
		for (int i = 0; i < 50000; i++) {
			double error = 0;
			error += nn.train(new double[] { 1, 0 }, 0);
			error += nn.train(new double[] { 0, 0 }, 1);
			error += nn.train(new double[] { 0, 1 }, 1);
			error += nn.train(new double[] { 1, 1 }, 1);
			if(i % 100 == 0) {
				System.out.println("--------");
				System.out.println("error: " + error);
				System.out.println("1 0 -> " + nn.compute(new double[] { 1, 0 }));
				System.out.println("0 0 -> " + nn.compute(new double[] { 0, 0 }));
				System.out.println("0 1 -> " + nn.compute(new double[] { 0, 1 }));
				System.out.println("1 1 -> " + nn.compute(new double[] { 1, 1 }));
			}
		}
		System.out.println("After 5000 * 4 tests\n--------");
		System.out.println("1 0 -> " + nn.compute(new double[] { 1, 0 }));
		System.out.println("0 0 -> " + nn.compute(new double[] { 0, 0 }));
		System.out.println("0 1 -> " + nn.compute(new double[] { 0, 1 }));
		System.out.println("1 1 -> " + nn.compute(new double[] { 1, 1 }));
	}
}
