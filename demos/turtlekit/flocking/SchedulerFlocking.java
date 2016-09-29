package turtlekit.flocking;

import java.io.FileWriter;
import java.io.IOException;

import jcuda.utils.Timer;
import turtlekit.kernel.TKScheduler;
import turtlekit.kernel.TurtleKit.Option;

public class SchedulerFlocking  extends TKScheduler {
	
	protected boolean writeFile = false;
	protected boolean logFile = false;
	
	@Override
	protected void activate() {
		super.activate();
		setSimulationDuration(20000);
	}

	
	@Override
	public void doSimulationStep() {
			
			Timer.startTimer(getTurtleActivator());
			getTurtleActivator().execute();
			Timer.stopTimer(getTurtleActivator());

			Timer.startTimer(getEnvironmentUpdateActivator());
			getEnvironmentUpdateActivator().execute();
			Timer.stopTimer(getEnvironmentUpdateActivator());
			
			getViewerActivator().execute();
			setGVT(getGVT() + 1);
	}

	@Override
	protected void end() {
		super.end();
		if (writeFile) {
			final String csvFile = "result";//getMadkitProperty("cvs.file");
			if (csvFile != null) {
				logger.info("pouet");
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
		}

		if (logFile) {
			getLogger().createLogFile();
		}
		logger.info("average agents"+ (Timer.getTimerValue(getTurtleActivator()) / 1000000 / getGVT())	);
		logger.info(getMadkitProperty(Option.envDimension.name()));
		logger.info("nb agents : "+getTurtleActivator().size());
		logger.info("Iteration : " + getGVT());
		logger.info(Timer.createPrettyString());
	}
}
