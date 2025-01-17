/*******************************************************************************
 * TurtleKit 3 - Agent Based and Artificial Life Simulation Platform
 * Copyright (C) 2011-2016 Fabien Michel
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

/**
 * A pheromone grid that uses a float array to store values. It is used for CPU
 * computations.
 */
public class CPUStreamPheromone extends AbstractPheromoneGrid<Float> {

	private final float[] values;
	private final float[] tmp;

	public CPUStreamPheromone(String name, TKEnvironment<?> environment, float evaporationCoeff, float diffusionCoeff) {
		super(name, environment, evaporationCoeff, diffusionCoeff);
		int gridSize = getWidth() * getHeight();
		values = new float[gridSize];
		tmp = new float[gridSize];
		setMaxEncounteredValue(Float.NEGATIVE_INFINITY);
	}

	@Override
	public Float get(int index) {
		return values[index];
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see turtlekit.pheromone.DataGrid#set(int, java.lang.Object)
	 */
	@Override
	public void set(int index, Float value) {
		values[index] = value;
	}

	public int getMaxDirection(int xcor, int ycor) {
//		return getMaxDirectionNoRandom(xcor, ycor);
		return getMaxDirectionRandomWhenEquals(xcor, ycor);
	}

	public int getMinDirection(int xcor, int ycor) {
//		return getMinDirectionNoRandom(xcor, ycor);
		return getMinDirectionRandomWhenEquals(xcor, ycor);
	}

	public int getMaxDirectionRandomWhenEquals(int xcor, int ycor) {
		int index = get1DIndex(xcor, ycor) * 8;
		return (int) GradientsCalculator.getMaxDirectionRandomWhenEquals(this, index);
	}

	public int getMaxDirectionNoRandom(int xcor, int ycor) {
		int index = get1DIndex(xcor, ycor) * 8;
		return (int) GradientsCalculator.getMaxDirectionNoRandom(this, index);
	}

	public int getMinDirectionRandomWhenEquals(int xcor, int ycor) {
		int index = get1DIndex(xcor, ycor) * 8;
		return (int) GradientsCalculator.getMinDirectionRandomWhenEquals(this, index);
	}

	public int getMinDirectionNoRandom(int xcor, int ycor) {
		int index = get1DIndex(xcor, ycor) * 8;
		return (int) GradientsCalculator.getMinDirectionNoRandom(this, index);
	}

	@Override
	public void updateValuesFromTmpGridKernel() {
		IntStream.range(0, values.length).parallel().forEach(i -> {
			values[i] += getTotalUpdateFromNeighbors(i * 8);
		});
	}

	public void diffusionAndEvaporationUpdateKernel() {
		float evapCoef = (float) getEvaporationCoefficient();
		IntStream.range(0, values.length).parallel().forEach(i -> {
			values[i] += getTotalUpdateFromNeighbors(i * 8);
			values[i] -= values[i] * evapCoef;
		});
	}

	@Override
	public void diffuseValuesToTmpGridKernel() {
		float diffCoef = (float) getDiffusionCoefficient();
		IntStream.range(0, values.length).parallel().forEach(i -> {
			float give = values[i] * diffCoef;
			values[i] -= give;
			tmp[i] = give / 8;
		});
	}

	@Override
	public void evaporationKernel() {
		float evapCoef = (float) getEvaporationCoefficient();
		if (evapCoef != 0) {
			IntStream.range(0, values.length).parallel().forEach(i -> values[i] -= values[i] * evapCoef);
		}
	}

	float getTotalUpdateFromNeighbors(int index) {
		return tmp[neighborsIndexes[index]] + tmp[neighborsIndexes[++index]] + tmp[neighborsIndexes[++index]]
				+ tmp[neighborsIndexes[++index]] + tmp[neighborsIndexes[++index]] + tmp[neighborsIndexes[++index]]
				+ tmp[neighborsIndexes[++index]] + tmp[neighborsIndexes[++index]];
	}

	@Override
	public void incValue(int x, int y, float quantity) {
		incValue(get1DIndex(x, y), quantity);
	}

	/**
	 * Adds <code>inc</code> to the current value of the cell with the corresponding
	 * index
	 * 
	 * @param index cell's index
	 * @param inc   how much to add
	 */
	public void incValue(int index, float inc) {
		set(index, inc + get(index));
	}

	/**
	 * Compute the maximum value in the grid
	 */
	@Override
	public void updateMaxValue() {

		double currentMax = IntStream.range(0, values.length).parallel().mapToDouble(i -> values[i]).max().getAsDouble();
		if (currentMax > getMaxEncounteredValue()) {
			setMaxEncounteredValue((float) currentMax);
			setLogMaxValue(Math.log10((float) getMaxEncounteredValue() + 1) / 256);
		}
	}

}
