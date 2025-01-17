/*******************************************************************************
 * TurtleKit 3 - Agent Based and Artificial Life Simulation Platform
 * Copyright (C) 2011-2014 Fabien Michel
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 ******************************************************************************/
package turtlekit.pheromone;

import java.util.stream.IntStream;

import turtlekit.kernel.TKEnvironment;

public class CPU_SobelPheromone extends CPUStreamPheromone {

	final double[] sobelGradientValues;

	public CPU_SobelPheromone(String name, TKEnvironment<?> env, float evaporationCoeff, float diffusionCoeff, int[] neighborsIndexes2) {
		super(name, env, evaporationCoeff, diffusionCoeff);
		sobelGradientValues = new double[getWidth() * getHeight()];
	}

//	private void computeSobel(int x, int y) {
//		float eValue = get(normeValue(x + 1, getWidth()), y);
//		float neValue = get(normeValue(x + 1, getWidth()), normeValue(y + 1, getHeight()));
//		float nValue = get(x, normeValue(y + 1, getHeight()));
//		float nwValue = get(normeValue(x - 1, getWidth()), normeValue(y - 1, getHeight()));
//		float wValue = get(normeValue(x - 1, getWidth()), y);
//		float swValue = get(normeValue(x - 1, getWidth()), normeValue(y - 1, getHeight()));
//		float sValue = get(x, normeValue(y - 1, getHeight()));
//		float seValue = get(normeValue(x + 1, getWidth()), normeValue(y - 1, getHeight()));
//		sobelGradientValues[get1DIndex(x, y)] = Math.toDegrees(Math.atan2(nwValue + 2 * nValue + neValue - seValue - 2 * sValue - swValue, // filter
//																																														// Y
//				neValue + 2 * eValue + seValue - swValue - 2 * wValue - nwValue // filter X
//		));
//	}

	@Override
	public int getMaxDirection(int xcor, int ycor) {
		return (int) sobelGradientValues[get1DIndex(xcor, ycor)];
	}

	@Override
	public int getMinDirection(int i, int j) {
		return (int) sobelGradientValues[get1DIndex(i, j)] + 180;
	}

	@Override
	public void applyDynamics() {
		super.applyDynamics();
		IntStream.range(0, sobelGradientValues.length).parallel().forEach(this::computeSobel);
	}

	private void computeSobel(int i) {
		int index = i * 8;
		float eValue = get(neighborsIndexes[index++]);
		float neValue = get(neighborsIndexes[index++]);
		float nValue = get(neighborsIndexes[index++]);
		float nwValue = get(neighborsIndexes[index++]);
		float wValue = get(neighborsIndexes[index++]);
		float swValue = get(neighborsIndexes[index++]);
		float sValue = get(neighborsIndexes[index++]);
		float seValue = get(neighborsIndexes[index]);
		sobelGradientValues[i] = Math
				.toDegrees(Math.atan2(nwValue + 2 * nValue + neValue - seValue - 2 * sValue - swValue, // filter
																																	// Y
						neValue + 2 * eValue + seValue - swValue - 2 * wValue - nwValue // filter X
				));
	}

}
