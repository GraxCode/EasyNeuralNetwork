package me.nov.neural;

import java.util.Random;

/**
 * This program is free software: 
 * you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, 
 * either version 3 of the License, or (at your option) any later version. 
 * This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
 * without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
 * You should have received a copy of the GNU General Public License along with this program. If not, see <http://www.gnu.org/licenses/>.
 * 
 * @author GraxCode
 */
public class NeuralNetwork {
	private static final int inputSize = 2;

	private static final int hiddenLayerSize = 16;
	private static final double learningRate = 0.05;

	private static final Random r = new Random();
	private double[][] weightsHL1;
	private double[][] weightsHL2;
	private double[] weightsO;

	public NeuralNetwork() {
		weightsHL1 = new double[hiddenLayerSize][inputSize + 1]; // and bias
		weightsHL2 = new double[hiddenLayerSize + 1][hiddenLayerSize + 1]; // and bias
		weightsO = new double[hiddenLayerSize + 1]; // and bias

		for (int i = 0; i < hiddenLayerSize; i++) {
			for (int j = 0; j < inputSize + 1; j++) {
				weightsHL1[i][j] = r.nextDouble() * 2d - 1;
			}
			for (int j = 0; j < hiddenLayerSize + 1; j++) {
				weightsHL2[i][j] = r.nextDouble() * 2d - 1;
				weightsO[i] = r.nextDouble() * 2d - 1;
			}
		}
	}

	public double compute(double[] input) {
		double[] hl1 = new double[hiddenLayerSize];

		for (int i = 0; i < hiddenLayerSize; i++) {
			double weightedValue = 0;
			for (int j = 0; j < inputSize; j++) {
				weightedValue += weightsHL1[i][j] * input[j];
			}
			weightedValue += weightsHL1[i][inputSize]; // bias

			hl1[i] = sigmoid(weightedValue);
		}
		double[] hl2 = new double[hiddenLayerSize];
		for (int i = 0; i < hiddenLayerSize; i++) {
			double weightedValue = 0;
			for (int j = 0; j < hiddenLayerSize; j++) {
				weightedValue += weightsHL2[i][j] * hl1[j];
			}
			weightedValue += weightsHL2[i][hiddenLayerSize]; // bias
			hl2[i] = sigmoid(weightedValue);
		}
		double weightedValue = 0;
		for (int j = 0; j < hiddenLayerSize; j++) {
			weightedValue += weightsO[j] * hl2[j];
		}
		weightedValue += weightsO[hiddenLayerSize]; // bias
		return sigmoid(weightedValue);
	}

	public double train(double[] input, double expected) {
		double[][] newWeightsHL1 = new double[hiddenLayerSize][inputSize + 1];
		double[][] newWeightsHL2 = new double[hiddenLayerSize][hiddenLayerSize + 1];
		double[] newWeightsO = new double[hiddenLayerSize + 1];

		// forwardpropagation
		double[] hl1 = new double[hiddenLayerSize];
		for (int i = 0; i < hiddenLayerSize; i++) {
			double weightedValue = 0;
			for (int j = 0; j < inputSize; j++) {
				weightedValue += weightsHL1[i][j] * input[j];
			}
			weightedValue += weightsHL1[i][inputSize]; // bias
			hl1[i] = sigmoid(weightedValue);
		}
		double[] hl2 = new double[hiddenLayerSize];
		for (int i = 0; i < hiddenLayerSize; i++) {
			double weightedValue = 0;
			for (int j = 0; j < hiddenLayerSize; j++) {
				weightedValue += weightsHL2[i][j] * hl1[j];
			}
			weightedValue += weightsHL2[i][hiddenLayerSize]; // bias
			hl2[i] = sigmoid(weightedValue);
		}
		double weightedValue = 0;
		for (int j = 0; j < hiddenLayerSize; j++) {
			weightedValue += weightsO[j] * hl2[j];
		}
		weightedValue += weightsO[hiddenLayerSize];// bias

		// backpropagation
		double output = sigmoid(weightedValue);
		double deltaOutput = output * (1d - output) * (expected - output);
		for (int i = 0; i < hiddenLayerSize; i++) {
			newWeightsO[i] = weightsO[i] + deltaOutput * hl2[i] * learningRate;
		}
		newWeightsO[hiddenLayerSize] = weightsO[hiddenLayerSize] + deltaOutput * learningRate; // bias

		// start with hl2
		double[] deltasHl2 = new double[hiddenLayerSize + 1];
		for (int i = 0; i < hiddenLayerSize; i++) {
			double deltaHl2 = hl2[i] * (1 - hl2[i]) * (deltaOutput * weightsO[i]);
			deltasHl2[i] = deltaHl2;
		}
		for (int i = 0; i < hiddenLayerSize; i++) {
			for (int j = 0; j < hiddenLayerSize; j++) {
				newWeightsHL2[i][j] = weightsHL2[i][j] + deltasHl2[i] * hl1[j] * learningRate;
			}
			newWeightsHL2[i][hiddenLayerSize] = weightsHL2[i][hiddenLayerSize] + deltasHl2[i] * learningRate; // bias
		}
		double[] deltasHl1 = new double[hiddenLayerSize];
		for (int i = 0; i < hiddenLayerSize; i++) {

			double sum = 0;
			for (int k = 0; k < hiddenLayerSize; k++) {
				sum += deltasHl2[k] * weightsHL2[k][i];
			}
			double deltaHl1 = hl1[i] * (1 - hl1[i]) * (sum);
			deltasHl1[i] = deltaHl1;
		}
		for (int i = 0; i < hiddenLayerSize; i++) {
			for (int j = 0; j < inputSize; j++) {
				newWeightsHL1[i][j] = weightsHL1[i][j] + deltasHl1[i] * input[j] * learningRate;
			}
			newWeightsHL1[i][inputSize] = weightsHL1[i][inputSize] + deltasHl1[i] * learningRate; // bias
		}

		weightsO = newWeightsO;
		weightsHL1 = newWeightsHL1;
		weightsHL2 = newWeightsHL2;
		return Math.abs(expected - output);
	}

	private double sigmoid(double value) {
		return 1d / (1d + Math.exp(-value));
	}
}
