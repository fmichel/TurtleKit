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

/**
 * @author Fabien Michel
 *
 */
public class DefaultCPUPheromoneGrid extends AbstractPheromoneGrid<Float> implements Pheromone<Float>{

	/**
	 * @param name
	 * @param width
	 * @param height
	 * @param evaporationCoeff
	 * @param diffusionCoeff
	 */
	
	final private float[] values;
	final private float[] tmp;
	final private int[] neighborsIndexes;

	
	public DefaultCPUPheromoneGrid(String name, int width, int height, float evaporationCoeff, float diffusionCoeff, int[] neighborsIndexes2) {
		super(name, width, height, evaporationCoeff, diffusionCoeff);
		values = new float[width * height];
		tmp = new float[width * height];
		neighborsIndexes = neighborsIndexes2;
		setMaximum(0f);
	}

	@Override
	public Float get(int index) {
		return values[index];
	}

	/* (non-Javadoc)
	 * @see turtlekit.pheromone.DataGrid#set(int, java.lang.Object)
	 */
	@Override
	public void set(int index, Float value) {
		values[index] = value;
	}
	

public int getMaxDirection(int xcor, int ycor) {
		float max = get(normeValue(xcor + 1, getWidth()), ycor);
		int maxDir = 0;

		float current = get(normeValue(xcor + 1, getWidth()), normeValue(ycor + 1, getHeight()));
		if (current > max) {
			max = current;
			maxDir = 45;
		}

		current = get(xcor, normeValue(ycor + 1, getHeight()));
		if (current > max) {
			max = current;
			maxDir = 90;
		}

		current = get(normeValue(xcor - 1, getWidth()), normeValue(ycor + 1, getHeight()));
		if (current > max) {
			max = current;
			maxDir = 135;
		}

		current = get(normeValue(xcor - 1, getWidth()), ycor);
		if (current > max) {
			max = current;
			maxDir = 180;
		}

		current = get(normeValue(xcor - 1, getWidth()), normeValue(ycor - 1, getHeight()));
		if (current > max) {
			max = current;
			maxDir = 225;
		}

		current = get(xcor, normeValue(ycor - 1, getHeight()));
		if (current > max) {
			max = current;
			maxDir = 270;
		}

		current = get(normeValue(xcor + 1, getWidth()), normeValue(ycor - 1, getHeight()));
		if (current > max) {

			max = current;
			maxDir = 315;
		}
		return maxDir;
	}

	public int getMinDirection(int i, int j) {
		float min = get(normeValue(i + 1, getWidth()), j);
		int minDir = 0;

		float current = get(normeValue(i + 1, getWidth()), normeValue(j + 1, getHeight()));
		if (current < min) {
			min = current;
			minDir = 45;
		}

		current = get(i, normeValue(j + 1, getHeight()));
		if (current < min) {
			min = current;
			minDir = 90;
		}

		current = get(normeValue(i - 1, getWidth()), normeValue(j + 1, getHeight()));
		if (current < min) {
			min = current;
			minDir = 135;
		}

		current = get(normeValue(i - 1, getWidth()), j);
		if (current < min) {
			min = current;
			minDir = 180;
		}

		current = get(normeValue(i - 1, getWidth()), normeValue(j - 1, getHeight()));
		if (current < min) {
			min = current;
			minDir = 225;
		}

		current = get(i, normeValue(j - 1, getHeight()));
		if (current < min) {
			min = current;
			minDir = 270;
		}

		current = get(normeValue(i + 1, getWidth()), normeValue(j - 1, getHeight()));
		if (current < min) {
			min = current;
			minDir = 315;
		}
		return minDir;
	}	
	
	

	@Override
	protected void diffusionUpdateKernel(){
        for (int i=getWidth()-1; i >=0 ; i--)
            for (int j=getHeight()-1; j >=0 ; j--){
            	values[get1DIndex(i,j)] += getTotalUpdateFromNeighbors(i, j);
            }
	}

	public void diffusionAndEvaporationUpdateKernel(){
		float evapCoef = getEvaporationCoefficient();
        for (int i=getWidth()-1; i >=0 ; i--)
            for (int j=getHeight()-1; j >=0 ; j--){
                int k = get1DIndex(i,j);
                values[k] += getTotalUpdateFromNeighbors(i, j);
				values[k] -= values[k] * evapCoef;
            }
	}

	 float getTotalUpdateFromNeighbors(int i, int j){
		 int index = get1DIndex(i, j)*8;

		return 
	        		tmp[neighborsIndexes[index]] +
	        tmp[neighborsIndexes[++index]] +
	        tmp[neighborsIndexes[++index]] +
	        tmp[neighborsIndexes[++index]] +
	        tmp[neighborsIndexes[++index]] +
	        tmp[neighborsIndexes[++index]] +
	        tmp[neighborsIndexes[++index]] +
	        tmp[neighborsIndexes[++index]];
	}
	 
	 @Override
		public void diffuseValuesToTmpGridKernel() {
	        float diffCoef = getDiffusionCoefficient();
			for (int i = values.length - 1; i >= 0 ; i--) {
				float give = values[i] * diffCoef;
				float giveToNeighbor = give / 8;
				values[i] -= give;//TODO a[k] = value - give
				tmp[i] = giveToNeighbor;
			}
		}


		@Override
		public void evaporationKernel() {
	        float evapCoef = (float) getEvaporationCoefficient();
				if (evapCoef != 0) {
					for (int i = values.length - 1; i >= 0; i--) {
						values[i] -= values[i] * evapCoef;
					}
				}
			}

		@Override
		public void incValue(int x, int y, float quantity) {
				incValue(get1DIndex(x, y), quantity);
			}

		/**
		 * Adds <code>inc</code> to the current value of the cell 
		 * with the corresponding index
		 * 
		 * @param index cell's index
		 * @param inc how much to add
		 */
		public void incValue(int index, float inc) {
//			inc += get(index);
//			if (inc > maximum)
//				setMaximum(inc);
//			set(index, inc);
			set(index, inc + get(index));
		}

}
