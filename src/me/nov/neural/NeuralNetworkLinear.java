package me.nov.neural;

import java.util.Random;

/**
 * This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version. This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details. You should have received a copy of the GNU General Public License along with this program. If not, see <http://www.gnu.org/licenses/>.
 * 
 * Activation function: ReLU
 * 
 * @author GraxCode
 * @version 2
 */
public class NeuralNetworkLinear {
	private static final int inputSize = 1;

	private static final int hiddenLayer1Size = 50;
	private static final int hiddenLayer2Size = 25;
	private static final double learningRate = 0.001;

	private static final Random r = new Random();
	public double[][] weightsHL1;
	public double[][] weightsHL2;
	public double[] weightsO;

	private static final double alpha = 0.01;

	public NeuralNetworkLinear() {
		weightsHL1 = new double[hiddenLayer1Size][inputSize + 1]; // and bias
		weightsHL2 = new double[hiddenLayer2Size + 1][hiddenLayer1Size + 1]; // and bias
		weightsO = new double[hiddenLayer2Size + 1]; // and bias
		for (int i = 0; i < hiddenLayer1Size; i++) {
			for (int j = 0; j < inputSize + 1; j++) {
				weightsHL1[i][j] = (r.nextDouble() * 2d - 1) * Math.sqrt(2d / (double) (inputSize + 1d));
			}
		}
		for (int i = 0; i < hiddenLayer2Size + 1; i++) {
			weightsO[i] = (r.nextDouble() * 2d - 1) *  Math.sqrt(2d / (double) (hiddenLayer2Size + 1d));
			for (int j = 0; j < hiddenLayer1Size + 1; j++) {
				weightsHL2[i][j] = (r.nextDouble() * 2d - 1) * Math.sqrt(2d / (double) (hiddenLayer1Size + 1d));
			}
		}
	}

	public strictfp double compute(double[] input) {
		if (input.length != inputSize) {
			throw new IllegalArgumentException();
		}
		double[] hl1 = new double[hiddenLayer1Size];

		for (int i = 0; i < hiddenLayer1Size; i++) {
			double weightedValue = 0;
			for (int j = 0; j < inputSize; j++) {
				weightedValue += weightsHL1[i][j] * input[j];
			}
			weightedValue += weightsHL1[i][inputSize]; // bias
			hl1[i] = relu(weightedValue);
		}
		double[] hl2 = new double[hiddenLayer2Size];
		for (int i = 0; i < hiddenLayer2Size; i++) {
			double weightedValue = 0;
			for (int j = 0; j < hiddenLayer1Size; j++) {
				weightedValue += weightsHL2[i][j] * hl1[j];
			}
			weightedValue += weightsHL2[i][hiddenLayer1Size]; // bias
			hl2[i] = relu(weightedValue);
		}
		double weightedValue = 0;
		for (int j = 0; j < hiddenLayer2Size; j++) {
			weightedValue += weightsO[j] * hl2[j];
		}
		weightedValue += weightsO[hiddenLayer2Size]; // bias
		return relu(weightedValue);
	}

	public strictfp double train(double[] input, double expected) {
		if (input.length != inputSize) {
			throw new IllegalArgumentException(String.valueOf(input.length));
		}
		double[][] newWeightsHL1 = new double[hiddenLayer1Size][inputSize + 1];
		double[][] newWeightsHL2 = new double[hiddenLayer2Size][hiddenLayer1Size + 1];
		double[] newWeightsO = new double[hiddenLayer2Size + 1];

		// forwardpropagation
		double[] hl1 = new double[hiddenLayer1Size];

		for (int i = 0; i < hiddenLayer1Size; i++) {
			double weightedValue = 0;
			for (int j = 0; j < inputSize; j++) {
				weightedValue += weightsHL1[i][j] * input[j];
			}
			weightedValue += weightsHL1[i][inputSize]; // bias

			hl1[i] = relu(weightedValue);
		}
		double[] hl2 = new double[hiddenLayer2Size];
		for (int i = 0; i < hiddenLayer2Size; i++) {
			double weightedValue = 0;
			for (int j = 0; j < hiddenLayer1Size; j++) {
				weightedValue += weightsHL2[i][j] * hl1[j];
			}
			weightedValue += weightsHL2[i][hiddenLayer1Size]; // bias
			hl2[i] = relu(weightedValue);
		}
		double weightedValue = 0;
		for (int j = 0; j < hiddenLayer2Size; j++) {
			weightedValue += weightsO[j] * hl2[j];
		}
		weightedValue += weightsO[hiddenLayer2Size]; // bias

		// backpropagation
		double output = relu(weightedValue);
		double deltaOutput = reluDerivative(output) * (expected - output);
		for (int i = 0; i < hiddenLayer2Size; i++) {
			newWeightsO[i] = weightsO[i] + deltaOutput * hl2[i] * learningRate;
		}
		newWeightsO[hiddenLayer2Size] = weightsO[hiddenLayer2Size] + deltaOutput * learningRate; // bias

		// start with hl2
		double[] deltasHl2 = new double[hiddenLayer2Size + 1];
		for (int i = 0; i < hiddenLayer2Size; i++) {
			deltasHl2[i] = reluDerivative(hl2[i]) * (deltaOutput * weightsO[i]);
		}
		for (int i = 0; i < hiddenLayer2Size; i++) {
			for (int j = 0; j < hiddenLayer1Size; j++) {
				newWeightsHL2[i][j] = weightsHL2[i][j] + deltasHl2[i] * hl1[j] * learningRate;
			}
			newWeightsHL2[i][hiddenLayer1Size] = weightsHL2[i][hiddenLayer1Size] + deltasHl2[i] * learningRate; // bias
		}
		double[] deltasHl1 = new double[hiddenLayer1Size];
		for (int i = 0; i < hiddenLayer1Size; i++) {
			double sum = 0;
			for (int k = 0; k < hiddenLayer2Size; k++) {
				sum += deltasHl2[k] * weightsHL2[k][i];
			}
			deltasHl1[i] = reluDerivative(hl1[i]) * (sum);
		}
		for (int i = 0; i < hiddenLayer1Size; i++) {
			for (int j = 0; j < inputSize; j++) {
				newWeightsHL1[i][j] = weightsHL1[i][j] + deltasHl1[i] * input[j] * learningRate;
			}
			newWeightsHL1[i][inputSize] = weightsHL1[i][inputSize] + deltasHl1[i] * learningRate; // bias
		}

		weightsO = newWeightsO;
		weightsHL1 = newWeightsHL1;
		weightsHL2 = newWeightsHL2;
		return Math.pow((expected - output), 2) / 2d;
	}

	private double relu(double value) {
		return Math.max(alpha * value, value);
	}

	private double reluDerivative(double value) {
		if (value <= 0) {
			return alpha;
		}
		return 1;
	}

	public static int getInputSize() {
		return inputSize;
	}

	public static int getHiddenLayer1Size() {
		return hiddenLayer1Size;
	}

	public static int getHiddenLayer2Size() {
		return hiddenLayer2Size;
	}

	public static double getLearningRate() {
		return learningRate;
	}

	public static double getAlpha() {
		return alpha;
	}
}
