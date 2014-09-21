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
package turtlekit.mle;

import turtlekit.cuda.CudaGPUGradientsPhero;
import turtlekit.cuda.CudaPheromone;
import turtlekit.kernel.TKEnvironment;
import turtlekit.pheromone.Pheromone;

public class MLEEnvironment extends TKEnvironment{
	
	@Override
	protected void activate() {
		super.activate();
		getPheromone(AbstractMLEAgent.PRE+"0", 20, 64); 
//		getPheromone(AbstractMLEAgent.ATT+"0", 0, 0); 
//		getPheromone(AbstractMLEAgent.REP+"0", 0, 0);

		getPheromone(AbstractMLEAgent.PRE+"1", 40, 50); 
		getPheromone(AbstractMLEAgent.ATT+"1", 87, 100); 
		getPheromone(AbstractMLEAgent.REP+"1",97, 100);
		
		getPheromone(AbstractMLEAgent.PRE+"2", 15, 100); 
		getPheromone(AbstractMLEAgent.ATT+"2", 86, 100); 
		getPheromone(AbstractMLEAgent.REP+"2",96, 100);
		
		getPheromone(AbstractMLEAgent.PRE+"3", 10, 100); 
		getPheromone(AbstractMLEAgent.ATT+"3", 85, 100); 
		getPheromone(AbstractMLEAgent.REP+"3", 92, 100);
//		
		getPheromone(AbstractMLEAgent.PRE+"4", 15, 100); 
		getPheromone(AbstractMLEAgent.ATT+"4", 84, 100); 
		getPheromone(AbstractMLEAgent.REP+"4",89, 100);

	}
	
//	@Override
//	protected void update() {
//		super.update();
//		for (Pheromone p : getPheromones()) {
//			((CudaPheromoneV3) p).updateV3();
//		}
//	}
	
	protected Pheromone createCudaPheromone(String name, int evaporationPercentage, int diffusionPercentage){
		if(GPU_GRADIENTS && ! name.contains("PRE"))
			return new CudaGPUGradientsPhero(name, getWidth(), getHeight(), evaporationPercentage, diffusionPercentage);
		return new CudaPheromone(name, getWidth(),	getHeight(), evaporationPercentage, diffusionPercentage);
	}
	

}
