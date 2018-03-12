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

import static turtlekit.kernel.TurtleKit.Option.envDimension;
import static turtlekit.kernel.TurtleKit.Option.environment;
import static turtlekit.kernel.TurtleKit.Option.scheduler;
import static turtlekit.kernel.TurtleKit.Option.startSimu;
import static turtlekit.kernel.TurtleKit.Option.turtles;
import static turtlekit.kernel.TurtleKit.Option.viewers;

import javax.swing.JOptionPane;

import turtlekit.kernel.TKLauncher;
import turtlekit.kernel.TurtleKit.Option;
import turtlekit.viewer.PheromoneViewer;

public class MLELauncher extends TKLauncher {
	
	@Override
	protected void activate() {
//		setMadkitProperty("GPU_gradients", "true");
		Object[] tab = { "50","100", "256","512","1024","1536","2048" };
		Object size = JOptionPane.showInputDialog(null,
				"environment size", "MLE world size",
				JOptionPane.QUESTION_MESSAGE, null, tab, "256");
		setMadkitProperty(envDimension, size.toString());
		Object[] tab2 = { "1", "5", "10", "20","30","40","50","60","70","80","90","100","120","140" };
		size = JOptionPane.showInputDialog(null, "MLE population density",
				"population density", JOptionPane.QUESTION_MESSAGE, null, tab2,
				"10");
		setMadkitProperty("popDensity", size.toString());
		initProperties();
		setMadkitProperty(startSimu, "true");
		setMadkitProperty(viewers, PheromoneViewer.class.getName());
		setMadkitProperty(scheduler, MLEScheduler.class.getName());
		setMadkitProperty(environment, MLEEnvironment.class.getName());
		setMadkitProperty(turtles, Particule.class.getName()+","+(int) Math.ceil(Double.parseDouble(getMadkitProperty("popDensity"))*getWidth()*getHeight()/100));
		super.activate();
	}

	
	public static void main(String[] args) {
		executeThisLauncher(
			"--popDensity",
			"100",
			Option.cuda.toString()
);
	}

}
