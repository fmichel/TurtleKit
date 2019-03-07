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

import java.awt.Component;
import java.awt.Container;
import java.io.File;
import java.io.IOException;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Properties;
import java.util.logging.Level;
import java.util.logging.Logger;

import javax.swing.Action;
import javax.swing.JCheckBoxMenuItem;
import javax.swing.JComponent;
import javax.swing.JMenu;
import javax.swing.JToggleButton;

import madkit.action.GUIManagerAction;
import madkit.action.GlobalAction;
import madkit.action.KernelAction;
import madkit.i18n.ErrorMessages;
import madkit.kernel.AbstractAgent;
import madkit.kernel.Madkit;
import madkit.kernel.Madkit.BooleanOption;
import madkit.kernel.MadkitOption;
import turtlekit.viewer.TKDefaultViewer;

public class TurtleKit extends AbstractAgent {
	
	private static final String	TK_LOGGER_NAME		= "[* TURTLEKIT *] ";

	
	static final List<String> tkDefaultMDKArgs = new ArrayList<>(
			Arrays.asList(Madkit.Option.configFile.toString(),
			"turtlekit/kernel/turtlekit.properties"));
	
	public static String VERSION ;
	
	public TurtleKit() {
	}
	
	@Override
	protected void activate() {
		setLogLevel(Level.parse(getMadkitProperty(LevelOption.turtleKitLogLevel)));
//		MadkitProperties p = getMadkitConfig();
//		p.list(System.err);
		final String[] urlsName = System.getProperty("java.class.path").split(File.pathSeparator);
		final String modelFile = getMadkitProperty(Option.model);
		if (modelFile != null) {
		    getLogger().info(() -> "----- Loading model "+modelFile);
			try {
				getMadkitConfig().loadPropertiesFromFile(modelFile);
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
		else if(getMadkitProperty(turtlekit.kernel.TurtleKit.Option.environment).equals(TKEnvironment.class.getName()) &&
				getMadkitProperty(turtlekit.kernel.TurtleKit.Option.viewers).equals(TKDefaultViewer.class.getName()) &&
				getMadkitProperty(turtlekit.kernel.TurtleKit.Option.launcher).equals(TKLauncher.class.getName()) &&
				getMadkitProperty(turtlekit.kernel.TurtleKit.Option.turtles).equals("null")
				){
				getLogger().fine(() -> "No valid configuration to start with: Desktop mode activated!");
			return;
		}
		launchAgent(getMadkitProperty(Option.launcher));
	}
	
	public TurtleKit(String... args) {
		main(args);
	}
	

	/**
	 * Builds a menu featuring the following actions:
	 * <ul>
	 * <li> {@link KernelAction#EXIT}
	 * <li> {@link KernelAction#COPY}
	 * <li> {@link KernelAction#RESTART}
	 * <li> {@link KernelAction#LAUNCH_NETWORK}
	 * <li> {@link KernelAction#STOP_NETWORK}
	 * <li> {@link GUIManagerAction#CONNECT_TO_IP}
	 * <li> {@link KernelAction#CONSOLE}
	 * <li> {@link KernelAction#LOAD_LOCAL_DEMOS}
	 * <li> {@link GUIManagerAction#LOAD_JAR_FILE}
	 * <li> {@link GUIManagerAction#ICONIFY_ALL}
	 * <li> {@link GUIManagerAction#DEICONIFY_ALL}
	 * <li> {@link GUIManagerAction#KILL_AGENTS}
	 * </ul>
	 * 
	 * @param agent the agent for which this menu
	 * will be built.
	 */
	public static void addTurleKitActionsTo(JComponent menuOrToolBar, AbstractAgent agent){
		try {//this bypasses class incompatibility
			final Method add = menuOrToolBar.getClass().getMethod("add", Action.class);
			final Method addSeparator = menuOrToolBar.getClass().getMethod("addSeparator");
			final Method addButton = Container.class.getMethod("add", Component.class);
			
			add.invoke(menuOrToolBar, turtleKitize(KernelAction.EXIT.getActionFor(agent)));
			addSeparator.invoke(menuOrToolBar);
			add.invoke(menuOrToolBar, turtleKitize(KernelAction.COPY.getActionFor(agent)));
			add.invoke(menuOrToolBar, turtleKitize(KernelAction.RESTART.getActionFor(agent)));
			addSeparator.invoke(menuOrToolBar);
			add.invoke(menuOrToolBar, GlobalAction.JCONSOLE);
			add.invoke(menuOrToolBar, KernelAction.CONSOLE.getActionFor(agent));
			if(menuOrToolBar instanceof JMenu){
				addButton.invoke(menuOrToolBar, new JCheckBoxMenuItem(GlobalAction.DEBUG));
			}
			else{
				final JToggleButton jToggleButton = new JToggleButton(GlobalAction.DEBUG);
				jToggleButton.setText(null);
				addButton.invoke(menuOrToolBar, jToggleButton);
			}
			addSeparator.invoke(menuOrToolBar);
			add.invoke(menuOrToolBar, GlobalAction.LOAD_JAR_FILE);
			addSeparator.invoke(menuOrToolBar);
			// not required now
//			add.invoke(menuOrToolBar, GUIManagerAction.ICONIFY_ALL.getActionFor(agent));
//			add.invoke(menuOrToolBar, GUIManagerAction.DEICONIFY_ALL.getActionFor(agent));
		} catch (IllegalAccessException | IllegalArgumentException e) {
			e.printStackTrace();
		} catch (SecurityException | InvocationTargetException | NoSuchMethodException e) {
		}
	}
	
	private static Action turtleKitize(Action a) {
		a.putValue(Action.SHORT_DESCRIPTION, a.getValue(Action.SHORT_DESCRIPTION).toString().replaceAll("MaDKit", "TurtleKit"));
		a.putValue(Action.LONG_DESCRIPTION, a.getValue(Action.LONG_DESCRIPTION).toString().replaceAll("MaDKit", "TurtleKit"));
		a.putValue(Action.NAME, a.getValue(Action.NAME).toString().replaceAll("MaDKit", "TurtleKit"));
		return a;
	}

	/**
	 * @param args
	 */
	public static void main(String[] args) {
//		JOptionPane.showMessageDialog(null,Arrays.deepToString(args));
		Collections.addAll(tkDefaultMDKArgs, args);
		if(tkDefaultMDKArgs.contains(Option.model.toString())){
			Collections.addAll(tkDefaultMDKArgs, BooleanOption.desktop.toString(),"false");
		}
//		final List<String> l = new ArrayList<>(
//				Arrays.asList(Madkit.Option.configFile.toString(),
//				"turtlekit/kernel/turtlekit.properties"));
//		if(args != null && args.length > 0){
//			l.addAll(Arrays.asList(args));
//			if(l.contains(Option.model.toString())){
//				l.add(BooleanOption.desktop.toString());
//				l.add("false");
//			}
//		}
//		JOptionPane.showMessageDialog(null,tkDefaultMDKArgs);
		new Madkit(tkDefaultMDKArgs.toArray(new String[tkDefaultMDKArgs.size()]));
//		new Madkit(l.toArray(new String[l.size()]));
	}
	
	/**
	 * TurtleKit options which are valued with a string representing parameters.
	 * These options could be used from the command line or using the main method of TurtleKit.
	 * 
	 * @author Fabien Michel
	 * 
	 */
	public enum Option implements MadkitOption {
		/**
		 * Used to launch turtles.
		 * This option can be used
		 * from the command line or using the main method of TurtleKit.
		 * 
		 * <pre>
		 * SYNOPSIS
		 * </pre>
		 * 
		 * <code><b>--turtles</b></code> TURTLE_CLASS_NAME[,NB][;OTHERS]
		 * <p>
		 * <ul>
		 * <li><i>TURTLE_CLASS_NAME</i>: the turtle class to launch</li>
		 * <li><i>NB</i> (integer optional): number of desired instances</li>
		 * </ul>
		 * 
		 * <pre>
		 * DESCRIPTION
		 * </pre>
		 * 
		 * The optional parameters could be used to (1) launch several different types of turtles and 
		 * (2) specify the number of desired instances of each
		 * type.
		 * 
		 * <pre>
		 * DEFAULT VALUE100000
		 * </pre>
		 * 
		 * Default value is <i>"null"</i>, meaning that no turtle has to be launched.
		 * <p>
		 * Default values for the optional parameters are
		 * <ul>
		 * <li><i>NB</i> : 1</li>
		 * </ul>
		 * 
		 * <pre>
		 * EXAMPLES
		 * </pre>
		 * 
		 * <ul>
		 * <li>--turtles myPackage.MyTurtle</li>
		 * <li>--turtles myPackage.MyTurtle,2000</li>
		 * <li>--turtles myPackage.MyTurtle,30000;other.OtherTurtle,40000</li>
		 * </ul>
		 */
		turtles,
		envWidth,
		envHeight,
		/**
		 * Should be used like this on the command line: --envDimension 100,100
		 */
		envDimension,
		/**
		 * Could be used to specify a .mdk properties 
		 * file that will be used to automatically launch
		 * a TK simulation.
		 */
		model, 
		/**
		 * Specify the name of the community which should be used.
		 * If not specified, the name of the model is used instead.
		 */
		community, 
		/**
		 * If set to <code>true</code>, the world is not wrapped.
		 * Default value is <code>false</code>
		 * 
		 */
		noWrap, 
		/**
		 * Specify the launcher agent class which has to be used.
		 */
		launcher,
		/**
		 * Specify the environment class which has to be used.
		 * Default is {@link Environment}
		 */
		environment,
		/**
		 * Specify the scheduler class which has to be used.
		 * Default is {@link TKScheduler}
		 */
		scheduler,
		/**
		 * If set to <code>true</code>, the simulation is started automatically.
		 * Default is {@link Environment}
		 * 
		 */
		startSimu,
		/**
		 * Specify the viewer classes to launch. If set to <code>null</code>, no viewer 
		 * will be launched. Several viewers could be set by separating their name with a ";"
		 */
		viewers, 
		/**
		 * Specify the patch class that should be used for creating the environment's grid. 
		 * Default is "turtlekit.kernel.Patch", that is {@link Patch}.
		 */
		patch, 
		/**
		 * If set to <code>true</code>, viewers will be launched
		 * in asynchronous display mode : faster viewing*/
		fastRendering,
		/**
		 * Set the rendering interval time, e.g. --renderingInterval 500
		 */
		renderingInterval,
		/**
		 * If set to <code>true</code>, TurtleKit will try to use Nvidia Cuda for 
		 * computing pheromones dynamics and random numbers, thus optimizing
		 * the simulation performances, especially when several pheromones are used.
		 * This requires the Cuda toolkit to be installed on the host.
		 * See {@link https://developer.nvidia.com/cuda-downloads} for more
		 * information
		 * 
		 */
		cuda,
		/**
		 * Specify the time at which the simulation should stop and quit
		 */
		endTime; 
		
		
		public String toString() {
			return "--"+name();
		};
	}
	
	/**
	 * MaDKit options valued with a string representing a {@link Level} value.
	 * These options could be used from the command line or using the main method of MaDKit.
	 * 
	 * @author Fabien Michel
	 * @since MaDKit 5.0.0.10
	 * @version 0.9
	 * 
	 */
	public enum LevelOption implements MadkitOption {
		/**
		 * Option defining the default agent log level for newly
		 * launched agents.
		 * Default value is "INFO". This value could be overridden
		 * individually by agents using {@link AbstractAgent#setLogLevel(Level)}.
		 * <p>
		 * Example:
		 * <ul>
		 * <li>--agentLogLevel OFF</li>
		 * <li>--agentLogLevel ALL</li>
		 * <li>--agentLogLevel FINE</li>
		 * </ul>
		 * 
		 * @see AbstractAgent#logger
		 * @see java.util.logging.Logger
		 * @see AbstractAgent#getMadkitProperty(String)
		 * @see AbstractAgent#setMadkitProperty(String, String)
		 */
		turtleLogLevel,
		/**
		 * Only useful for TK kernel developers
		 */
		tkKernelLogLevel,
		/**
		 * log level for the {@link TKEnvironment} agent
		 */
		environmentLogLevel,
		/**
		 * Only useful for TK desktop developers
		 */
		tkGuiLogLevel,
		/**
		 * Can be used to make TK more quiet
		 */
		turtleKitLogLevel;

		Level getValue(Properties session) {
			try {
				return Level.parse(session.getProperty(name()));
			} catch (IllegalArgumentException e) {//TODO simplify this log
				Logger.getLogger(TK_LOGGER_NAME).log(Level.SEVERE,
						ErrorMessages.OPTION_MISUSED.toString() + " " + name() + " : " + session.getProperty(name()), e);
			}
			return Level.ALL;
		}

		/**
		 * Returns the constant's name prefixed by "<code>--</code>" so that
		 * it could interpreted as an option of the command line or
		 * in {@link Madkit#Madkit(String...)}.
		 */
		@Override
		public String toString() {
			return "--" + name();
		}
	}


}
