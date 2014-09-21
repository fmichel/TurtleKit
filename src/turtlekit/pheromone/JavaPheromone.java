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


public class JavaPheromone extends Pheromone {

	final float[] values;
	final float[] tmp;
	final private int[] neighborsIndexes;
	
	public JavaPheromone(String name, int width, int height, int evaporationPercentage, int diffusionPercentage, int[] neighborsIndexes2) {
		this(name, width, height, evaporationPercentage / 100f, diffusionPercentage / 100f, neighborsIndexes2);
	}

	public JavaPheromone(String name, int width, int height, float evaporationPercentage, float diffusionPercentage, int[] neighborsIndexes2) {
		super(name, width, height, evaporationPercentage, diffusionPercentage);
		values = new float[width*height];
		tmp = new float[width*height];
		neighborsIndexes = neighborsIndexes2;
	}

	/* (non-Javadoc)
	 * @see pheromone.Pheromone#getValue(int)
	 */
	@Override
	public float get(int index) {
		return values[index];
	}
	
	public FloatBuffer getValuesFloatBuffer(){
		return FloatBuffer.allocate(width*height).put(values);
	}
	

	@Override
	public void set(int index, float value) {
		values[index] = value;
		if(value > getMaximum())
			setMaximum(value);
	}

	private void diffusionUpdate(){
        for (int i=getWidth()-1; i >=0 ; i--)
            for (int j=getHeight()-1; j >=0 ; j--){
            	values[get1DIndex(i,j)] += getTotalUpdateFromNeighbors(i, j);
            }
	}

	private void diffusionUpdateThenEvaporation(){
		float evapCoef = getEvaporationCoefficient();
        for (int i=getWidth()-1; i >=0 ; i--)
            for (int j=getHeight()-1; j >=0 ; j--){
                int k = get1DIndex(i,j);
                values[k] += getTotalUpdateFromNeighbors(i, j);
				values[k] -= values[k] * evapCoef;
            }
	}

//	 float getTotalUpdateFromNeighbors(int i, int j){
//	        return 
//	        tmp[get1DIndex(i, normeValue(j-1,height))] +
//			tmp[get1DIndex(i, normeValue(j+1,height))] +
//	        tmp[get1DIndex(normeValue(i-1,width), normeValue(j-1,height))] +
//	        tmp[get1DIndex(normeValue(i-1,width), j)] +
//	        tmp[get1DIndex(normeValue(i-1,width), normeValue(j+1,height))] +
//	        tmp[get1DIndex(normeValue(i+1,width), normeValue(j-1,height))] +
//	        tmp[get1DIndex(normeValue(i+1,width), j)] +
//	        tmp[get1DIndex(normeValue(i+1,width), normeValue(j+1,height))];
//	}
	 
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
	 
	 int normeValue(int x, int width){
			if(x < 0) //-1
				return width - 1;
			if(x == width)
				return 0;
			return x;
		}

	@Override
	public void diffusion() {
			diffuseValues();
			diffusionUpdate();	
//            if (env.wrap) {
//                for (int i=getWidth()-1; i >=0 ; i--)
//                    for (int j=getHeight()-1; j >=0 ; j--){
//                    	final int index = getIndex(i, j);
//                    	tmp[index] = (values[index] * diffCoef) / 8;
//                        }
//                for (int i=getWidth()-1; i >=0 ; i--)
//                    for (int j=getHeight()-1; j >=0 ; j--){
//                        }
//                    }
//            } else
//                for (int i=env.x-1; i >=0 ; i--)
//                    for (int j=env.y-1; j >=0 ; j--){
//                env.grid[i][j].diffusion=gridValues[i][j]*(diffCoef/env.grid[i][j].neighbors.length);
//                gridValues[i][j]-=gridValues[i][j]*diffCoef;
//                    }
//            for (int i=env.x-1; i >=0 ; i--)
//                for (int j=env.y-1; j >=0 ; j--){
//                    final Patch[] p=env.grid[i][j].neighbors;
//                    for (int a=p.length-1;a>=0;a--)
//                        gridValues[i][j]+=p[a].diffusion;
//                }
//        }
	}


	private void diffuseValues() {
        float diffCoef = getDiffusionCoefficient();
		for (int i = values.length - 1; i >= 0 ; i--) {
			float give = values[i] * diffCoef;
			float giveToNeighbor = give / 8;
			values[i] -= give;//TODO a[k] = value - give
			tmp[i] = giveToNeighbor;
		}
	}


	@Override
	public void diffusionAndEvaporation() {
			diffuseValues();
			diffusionUpdateThenEvaporation();
	}


	@Override
	public void evaporation() {
        float evapCoef = (float) getEvaporationPercentage().getValue() / 100;
			if (evapCoef != 0) {
				for (int i = values.length - 1; i >= 0; i--) {
					values[i] -= values[i] * evapCoef;
				}
			}
		}

	public int[] getNeighborsIndexes() {
		return neighborsIndexes;
	}
}
