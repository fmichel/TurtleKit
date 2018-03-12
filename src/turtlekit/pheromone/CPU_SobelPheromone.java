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

public class CPU_SobelPheromone extends DefaultCPUPheromoneGrid {

    final double[] sobelGradientValues;

    public CPU_SobelPheromone(String name, int width, int height, float evaporationCoeff, float diffusionCoeff, int[] neighborsIndexes2) {
	super(name, width, height, evaporationCoeff, diffusionCoeff, neighborsIndexes2);
	sobelGradientValues = new double[width * height];
    }

    private void computeSobel(int x, int y) {
	float eValue = get(normeValue(x + 1, getWidth()), y);
	float neValue = get(normeValue(x + 1, getWidth()), normeValue(y + 1, getHeight()));
	float nValue = get(x, normeValue(y + 1, getHeight()));
	float nwValue = get(normeValue(x - 1, getWidth()), normeValue(y - 1, getHeight()));
	float wValue = get(normeValue(x - 1, getWidth()), y);
	float swValue = get(normeValue(x - 1, getWidth()), normeValue(y - 1, getHeight()));
	float sValue = get(x, normeValue(y - 1, getHeight()));
	float seValue = get(normeValue(x + 1, getWidth()), normeValue(y - 1, getHeight()));
	sobelGradientValues[get1DIndex(x, y)] = Math.toDegrees(Math.atan2(
		nwValue + 2 * nValue + neValue - seValue - 2 * sValue - swValue, // filter Y
		neValue + 2 * eValue + seValue - swValue - 2 * wValue - nwValue // filter X
	));
    }

    @Override
    public int getMaxDirection(int xcor, int ycor) {
	return (int) sobelGradientValues[get1DIndex(xcor, ycor)];
    }

    @Override
    public int getMinDirection(int i, int j) {
	return (int) sobelGradientValues[get1DIndex(i, j)] + 180;
    }

    @Override
    public void diffusionAndEvaporation() {
	super.diffusionAndEvaporation();
	for (int i = getWidth() - 1; i >= 0; i--)
	    for (int j = getHeight() - 1; j >= 0; j--) {
		computeSobel(i, j);
	    }
    }

}
