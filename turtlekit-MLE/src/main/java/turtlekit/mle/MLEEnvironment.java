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
import turtlekit.pheromone.PheroColorModel;
import turtlekit.pheromone.PheroColorModel.MainColor;
import turtlekit.pheromone.Pheromone;

public class MLEEnvironment extends TKEnvironment{
	
	@Override
	protected void onActivation() {
		super.onActivation();
		Pheromone<?> p = getPheromone(MLEAgent.PRE + "0", 0.20f, 0.60f);
		p.setColorModel(PheroColorModel.RED);
//		p.setColorModel(new PheroColorModel(0, 0, 0, MainColor.RED));

		p = getPheromone(MLEAgent.PRE + "1", 0.40f, 0.50f);
		p.setColorModel(new PheroColorModel(50, 50, 50, MainColor.GREEN));

		p = getPheromone(MLEAgent.ATT + "1", 0.87f, 0.60f);
		p.setColorModel(new PheroColorModel(50, 50, 50, MainColor.GREEN));

		p = getPheromone(MLEAgent.REP + "1", 0.97f, 0.60f);
		p.setColorModel(new PheroColorModel(100, 50, 50, MainColor.GREEN));

		p = getPheromone(MLEAgent.PRE + "2", 0.15f, 0.60f);
		p.setColorModel(new PheroColorModel(100, 100, 100, MainColor.RED));

		p = getPheromone(MLEAgent.ATT + "2", 0.86f, 0.60f);
		p.setColorModel(new PheroColorModel(100, 100, 100, MainColor.RED));

		p = getPheromone(MLEAgent.REP + "2", 0.96f, 0.60f);
		p.setColorModel(new PheroColorModel(100, 100, 100, MainColor.RED));

		p = getPheromone(MLEAgent.PRE + "3", 0.10f, 1.00f);
		p.setColorModel(new PheroColorModel(50, 50, 50, MainColor.BLUE));
		p = getPheromone(MLEAgent.ATT + "3", 0.85f, 1.00f);
		p.setColorModel(new PheroColorModel(50, 50, 50, MainColor.BLUE));

		p = getPheromone(MLEAgent.REP + "3", 0.92f, 1.00f);
		p.setColorModel(new PheroColorModel(100, 50, 50, MainColor.BLUE));

		//
		p = getPheromone(MLEAgent.PRE + "4", 0.15f, 1.00f);
		p.setColorModel(new PheroColorModel(150, 50, 50, MainColor.RED));

		p = getPheromone(MLEAgent.ATT + "4", 0.84f, 1.00f);
		p.setColorModel(new PheroColorModel(150, 50, 50, MainColor.RED));

		p = getPheromone(MLEAgent.REP + "4", 0.89f, 1.00f);
		p.setColorModel(new PheroColorModel(150, 50, 50, MainColor.RED));

	}
	
	protected Pheromone createCudaPheromone(String name, float evaporationCoefficient, float diffusionCoefficient) {
		if (GPU_GRADIENTS && !name.contains("PRE")) {
			return new CudaGPUGradientsPhero(name, this, evaporationCoefficient, diffusionCoefficient);
		}
		return new CudaPheromone(name, this, evaporationCoefficient, diffusionCoefficient);
	}

}
