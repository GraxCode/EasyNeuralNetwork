package me.nov.neural;

import java.awt.Color;
import java.awt.Font;
import java.awt.Graphics;
import java.util.Random;

import javax.swing.JFrame;
import javax.swing.JPanel;
/**
 * This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version. This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details. You should have received a copy of the GNU General Public License along with this program. If not, see <http://www.gnu.org/licenses/>.
 * 
 * A training simulation for both networks
 * 
 * @author GraxCode
 * @version 2
 */
@SuppressWarnings("serial")
public class SineFunctionComparison extends JFrame {
	private static NeuralNetworkLinear nnl = new NeuralNetworkLinear();
	private static NeuralNetwork nns = new NeuralNetwork();

	private static Random r = new Random();
	private static int iter = 0;

	public SineFunctionComparison() {
		int size = 800;
		double half = size / 2d;
		double divisor = half / Math.PI;
		setBounds(100, 100, size + 20, size + 40);
		this.setContentPane(new JPanel() {

			@SuppressWarnings("static-access")
			@Override
			protected void paintComponent(Graphics g) {
				g.setColor(Color.red);
				int lastPoint = (int) (half - (nnl.compute(new double[] { 0 }) - 0.5d) * 2d * half);
				for (int d = 1; d < size; d++) {
					int point = (int) (half - (nnl.compute(new double[] { d / divisor }) - 0.5d) * 2d * half);
					g.drawLine(d - 1, lastPoint, d, point);
					lastPoint = point;
				}
				g.setColor(Color.magenta);
				lastPoint = (int) (half - (nns.compute(new double[] { 0 }) - 0.5d) * 2d * half);
				for (int d = 1; d < size; d++) {
					int point = (int) (half - (nns.compute(new double[] { d / divisor }) - 0.5d) * 2d * half);
					g.drawLine(d - 1, lastPoint, d, point);
					lastPoint = point;
				}
				g.setColor(Color.gray);
				lastPoint = (int) (half - Math.sin(0) * half);
				for (int d = 1; d < size; d++) {
					int point = (int) (half - Math.sin(d / divisor) * half);
					g.drawLine(d - 1, lastPoint, d, point);
					lastPoint = point;
				}
				g.setFont(new Font("TimesRoman", Font.PLAIN, 12));
				g.setColor(Color.blue);

				g.drawString("Iterations: " + iter, 10, 20);
				g.setColor(Color.red);
				g.drawString("ReLU Error: " + getErrorReLU(), 10, 40);
				g.drawString("ReLU LR: " + nnl.getLearningRate(), 10, 60);
				g.setColor(Color.magenta);
				g.drawString("Sigmoid Error: " + getErrorSigmoid(), 10, 80);
				g.drawString("Sigmoid LR: " + nns.getLearningRate(), 10, 100);
			}
		});
		new Thread(() -> {
			while (true) {
				try {
					// train more that actually used for better results
					double i = (r.nextDouble() - 0.5d) * Math.PI * 8;
					nnl.train(new double[] { i }, (Math.sin(i) / 2d) + 0.5);
					nns.train(new double[] { i }, (Math.sin(i) / 2d) + 0.5);
					iter++;
					repaint();
				} catch (Exception e) {
					e.printStackTrace();
				}
			}
		}).start();
		this.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		this.setTitle("Comparison of ReLU and Sigmoid Neural Networks");
	}

	protected double getErrorReLU() {
		double accuracy = 1000;
		double sum = 0;
		for (double i = 1; i < accuracy; i++) {
			double pos = ((i / accuracy) - 0.5d) * Math.PI * 8;
			sum += Math.pow(Math.sin(pos) - nnl.compute(new double[] { pos }), 2);
		}
		return Math.round(sum / accuracy * 10000d) / 10000d;
	}

	protected double getErrorSigmoid() {
		double accuracy = 1000;
		double sum = 0;
		for (double i = 1; i < accuracy; i++) {
			double pos = ((i / accuracy) - 0.5d) * Math.PI * 8;
			sum += Math.pow(Math.sin(pos) - nns.compute(new double[] { pos }), 2);
		}
		return Math.round(sum / accuracy * 10000d) / 10000d;
	}

	public static void main(String[] args) {
		new SineFunctionComparison().setVisible(true);
	}

}
