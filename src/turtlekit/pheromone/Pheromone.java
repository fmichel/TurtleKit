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

import java.nio.FloatBuffer;

public abstract class Pheromone {

	final int width, height;
	private float maximum = 0;
	private PheromoneCoefficientBoundedRangeModel diffusionPercentage,
			evaporationPercentage;
	private String name;
	private PheromoneView defaultView;

    public Pheromone(String name, int width, int height, final int evapPercentage,
			final int diffPercentage
			) {
    	this(name,width,height,evapPercentage / 100f, diffPercentage / 100f);
	}

    public Pheromone(String name, int width, int height, final float evaporationCoeff,
			final float diffusionCoeff
			) {
		this.width = width;
		this.height = height;
		this.name = name;
		diffusionPercentage = new PheromoneCoefficientBoundedRangeModel(diffusionCoeff);
		evaporationPercentage = new PheromoneCoefficientBoundedRangeModel(evaporationCoeff);
	}
    
	protected int get1DIndex(int x, int y) {
		return y * width + x;
	}
	
//	/**
//	 * @return the diffusion coefficient as a float between 0 and 1, e.g. 0.33 for 33% 
//	 */
//	final public float getDiffusionCoefficient(){
//		return (float) getDiffusionPercentage().getValue() / 100;
//	}

	/**
	 * @return the diffusion coefficient as a float between 0 and 1, e.g. 0.33 for 33% 
	 */
	final public float getDiffusionCoefficient(){
		return diffusionPercentage.getCoefficient();
	}

	/**
	 * @return the evaporation coefficient as a float between 0 and 1, e.g. 0.33 for 33% 
	 */
	final public float getEvaporationCoefficient(){
		return evaporationPercentage.getCoefficient();
	}

	final public float get(int x, int y) {
		return get(get1DIndex(x, y));
	}

	public void set(int x, int y, float value) {
		if (value > maximum)
			maximum = value;
		set(get1DIndex(x, y), value);
	}

	public void incValue(int xcor, int ycor, float value) {
		incValue(get1DIndex(xcor, ycor), value);
	}

	public abstract FloatBuffer getValuesFloatBuffer();

	/**
	 * access to underlying buffer
	 * 
	 * @param index
	 * @return
	 */
	public abstract float get(int index);

	/**
	 * access to underlying buffer
	 * 
	 * @param index
	 * @return
	 */
	protected abstract void set(int index, float value);

//	public int getEvaporationPercentage() {
//		return evaporationPercentage.getValue();
//	}

	@Deprecated
	public void setEvaporationPercentage(int evaporationPercentage) {
		this.evaporationPercentage.setCoefficient(evaporationPercentage / 100f);
	}

//	/**
//	 * @param diffusionPercentage an int between 0 and 100
//	 */
//	public void setDiffusionPercentage(int diffusionPercentage) {
//		this.diffusionPercentage.setValue(diffusionPercentage);
//	}

	/**
	 * @return the width
	 */
	public int getWidth() {
		return width;
	}

	/**
	 * @return the height
	 */
	public int getHeight() {
		return height;
	}

	public abstract void diffusion();

	public abstract void diffusionAndEvaporation();

	public abstract void evaporation();

	/**
	 * @return the maximum
	 */
	public float getMaximum() {
		return maximum;
	}

	public void updateFieldMaxDir() {
}

	/**
	 * @param maximum
	 *            the maximum to set
	 */
	public void setMaximum(float maximum) {
		this.maximum = maximum;
	}

	@Override
	public String toString() {
		return getClass().getSimpleName()+" "+ name + " evap=" + getEvaporationPercentage().getValue() + " diff=" + getDiffusionPercentage().getValue()	+ " max = " + getMaximum();
	}

	int normeValue(int x, int width) {
		if (x < 0) // -1
			return width - 1;
		if (x == width)
			return 0;
		return x;
	}

	public int getMaxDirection(int xcor, int ycor) {
		float max = get(normeValue(xcor + 1, width), ycor);
		int maxDir = 0;

		float current = get(normeValue(xcor + 1, width), normeValue(ycor + 1, height));
		if (current > max) {
			max = current;
			maxDir = 45;
		}

		current = get(xcor, normeValue(ycor + 1, height));
		if (current > max) {
			max = current;
			maxDir = 90;
		}

		current = get(normeValue(xcor - 1, width), normeValue(ycor + 1, height));
		if (current > max) {
			max = current;
			maxDir = 135;
		}

		current = get(normeValue(xcor - 1, width), ycor);
		if (current > max) {
			max = current;
			maxDir = 180;
		}

		current = get(normeValue(xcor - 1, width), normeValue(ycor - 1, height));
		if (current > max) {
			max = current;
			maxDir = 225;
		}

		current = get(xcor, normeValue(ycor - 1, height));
		if (current > max) {
			max = current;
			maxDir = 270;
		}

		current = get(normeValue(xcor + 1, width), normeValue(ycor - 1, height));
		if (current > max) {
			max = current;
			maxDir = 315;
		}
		return maxDir;
	}

	public int getMinDirection(int i, int j) {
		float min = get(normeValue(i + 1, width), j);
		int minDir = 0;

		float current = get(normeValue(i + 1, width), normeValue(j + 1, height));
		if (current < min) {
			min = current;
			minDir = 45;
		}

		current = get(i, normeValue(j + 1, height));
		if (current < min) {
			min = current;
			minDir = 90;
		}

		current = get(normeValue(i - 1, width), normeValue(j + 1, height));
		if (current < min) {
			min = current;
			minDir = 135;
		}

		current = get(normeValue(i - 1, width), j);
		if (current < min) {
			min = current;
			minDir = 180;
		}

		current = get(normeValue(i - 1, width), normeValue(j - 1, height));
		if (current < min) {
			min = current;
			minDir = 225;
		}

		current = get(i, normeValue(j - 1, height));
		if (current < min) {
			min = current;
			minDir = 270;
		}

		current = get(normeValue(i + 1, width), normeValue(j - 1, height));
		if (current < min) {
			min = current;
			minDir = 315;
		}
		return minDir;
	}

	/**
	 * Adds <code>inc</code> to the current value of the cell 
	 * with the corresponding index
	 * 
	 * @param index cell's index
	 * @param inc how much to add
	 */
	public void incValue(int index, float inc) {
//		inc += get(index);
//		if (inc > maximum)
//			setMaximum(inc);
//		set(index, inc);
		set(index, inc + get(index));
	}

	public String getName() {
		return name;
	}

	/**
	 * @return the diffusionPercentage
	 */
	public PheromoneCoefficientBoundedRangeModel getDiffusionPercentage() {
		return diffusionPercentage;
	}

	/**
	 * @return the evaporationPercentage
	 */
	public PheromoneCoefficientBoundedRangeModel getEvaporationPercentage() {
		return evaporationPercentage;
	}


}
