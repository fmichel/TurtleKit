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

import java.io.FileWriter;
import java.io.IOException;

import jcuda.utils.Timer;
import turtlekit.kernel.TKScheduler;
import turtlekit.kernel.TurtleKit.Option;

public class MLEScheduler extends TKScheduler {
	
	@Override
	protected void activate() {
		super.activate();
	}

	
	@Override
	public void doSimulationStep() {
			if(getGVT() == 100)
				Particule.MUTATION=true;
//			Timer.startTimer(this);
			
			Timer.startTimer(getTurtleActivator());
			getTurtleActivator().execute();
			Timer.stopTimer(getTurtleActivator());

			Timer.startTimer(getEnvironmentUpdateActivator());
			getEnvironmentUpdateActivator().execute();
			Timer.stopTimer(getEnvironmentUpdateActivator());
			
//			Timer.stopTimer(this);
			
			getViewerActivator().execute();
			setGVT(getGVT() + 1);
	}

	@Override
	protected void end() {
		super.end();
		final String csvFile = getMadkitProperty("cvs.file");
		if (csvFile != null) {
			final String envSize = getMadkitProperty(Option.envHeight.name());
			int size = Integer.parseInt(envSize);
			size = size * size;
			String results = envSize;
			results += ";" + Timer.getAverageTimerValue(this);
			results += ";" + Timer.getAverageTimerValue(getTurtleActivator());
			results += ";"
					+ Timer.getAverageTimerValue(getEnvironmentUpdateActivator());
			results += ";" + (getTurtleActivator().size() * 100 / size) + "%";
			results += ";" + getTurtleActivator().size() + "\n";
			try (FileWriter fw = new FileWriter(csvFile, true)) {
				System.err.println(results);
				fw.write(results);
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
		logger.info("average agents"+ (Timer.getTimerValue(getTurtleActivator()) / 1000000 / getGVT())	);
		//		getLogger().createLogFile();
		logger.info(getMadkitProperty(Option.envDimension.name()));
		logger.info("nb agents : "+getTurtleActivator().size());
		logger.info(Timer.createPrettyString());
	}

}
