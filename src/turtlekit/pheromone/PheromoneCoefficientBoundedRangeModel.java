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

import javax.swing.DefaultBoundedRangeModel;

public class PheromoneCoefficientBoundedRangeModel extends DefaultBoundedRangeModel {

	/**
	 * 
	 */
	private static final long serialVersionUID = 3828277289372933897L;

	public PheromoneCoefficientBoundedRangeModel(float initialValue) {
		super((int) (initialValue * getPrecision(initialValue)),0,0,getPrecision(initialValue));
	}
	

	private static int getPrecision(Float f){
		if(f < 0 || f > 1)
			throw new IllegalArgumentException("coeff should be between 0 and 1");
		final String s = f.toString();
		final int pow = (int) Math.pow(10, s.length() - s.indexOf('.') - 1);
		return pow > 10 ? pow : 100;
	}
	
	/**
	 * @return the coefficient as a float between 0 and 1 
	 */
	public float getCoefficient(){
		return (float) getValue() / getMaximum();
	}
	
	public void setCoefficient(float coeff) {
		final int precision = getPrecision(coeff);
		if(precision > getMaximum())
			setMaximum(precision);
		super.setValue((int) (coeff * precision));
	}

	

}
