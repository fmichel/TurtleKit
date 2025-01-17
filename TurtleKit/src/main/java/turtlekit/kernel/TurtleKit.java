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
package turtlekit.kernel;

import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;
import java.util.ArrayList;
import java.util.List;
import java.util.logging.Level;
import java.util.stream.IntStream;

import madkit.kernel.Agent;
import madkit.kernel.MadkitClassLoader;
import madkit.simulation.EngineAgents;
import madkit.simulation.SimuLauncher;
import turtlekit.cuda.CudaPlatform;
import turtlekit.viewer.TKViewer;

@EngineAgents(scheduler = TKScheduler.class, environment = TKEnvironment.class, viewers = { TKViewer.class })
public class TurtleKit extends SimuLauncher {

	private static final String TK_LOGGER_NAME = "[* TURTLEKIT *] ";

	// static final List<String> tkDefaultMDKArgs = new
	// ArrayList<>(Arrays.asList(Madkit.Option.configFile.toString(),
	// "turtlekit/kernel/turtlekit.properties"));

	public static String VERSION;

	@Override
	protected void onActivation() {
//		getLogger().setLevel(getKernelConfiguration().getLevel("tkLogLevel"));

		if (getKernelConfig().getBoolean("headless")) {
			getKernelConfig().setProperty("start", true);
		}
		if (getKernelConfig().getBoolean("cuda")) {
			CudaPlatform.init(Level.ALL);
			getKernelConfig().setProperty("cuda", CudaPlatform.isCudaAvailable());
		}
		getLogger().info(" -------- TurtleKit starting -------- ");
		super.onActivation();
	}

//	@Override
//	protected void createSimulationInstance() {
//		super.createSimulationInstance();
//		if (getKernelConfig().getBoolean("start")) {
//			startSimulation();
//		}
//	}

	@Override
	protected void onLaunchSimulatedAgents() {
		List<String> turtlesToLaunch = getKernelConfig().getList(String.class, "turtles");
		if (turtlesToLaunch == null)
			return;
		for (String turtles : turtlesToLaunch) {// TODO parallel
			String[] classAndNb = turtles.split(",");
			String className = classAndNb[0].trim();
			try {
				Class<?> turtleClass = MadkitClassLoader.getLoader().loadClass(className);
//				getEnvironment().addStaticParametersToModelFrom(turtleClass);
				final Constructor<?> agentClass = turtleClass.getConstructor();
				int nb = Integer.parseInt(classAndNb[1]);
				IntStream.rangeClosed(1, nb).sequential().forEach(i -> {
					try {
						launchAgent((Agent) agentClass.newInstance());
					} catch (InstantiationException | IllegalAccessException | IllegalArgumentException
							| InvocationTargetException e) {
						e.printStackTrace();
					}
				});
			} catch (ClassNotFoundException | NoSuchMethodException | SecurityException | IllegalArgumentException e) {
				e.printStackTrace();
			}
		}
	}

	/**
	 * This offers a convenient way to create a main method that launches a
	 * simulation using the launcher class under development. This call only works
	 * in the main method of the launcher.
	 *
	 * @param args MaDKit or TurtleKit options
	 * @since TurtleKit 3.0.0.1
	 */
	protected static void executeThisLauncher(String... args) {
		final List<String> arguments = new ArrayList<>(List.of(args));
		String className = Utils.getClassFromMainStackTrace();
		arguments.addAll(List.of("-la", className));
		TutleKit4.main(arguments.toArray(new String[arguments.size()]));
	}

}
