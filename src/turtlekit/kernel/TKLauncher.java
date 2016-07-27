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

import static turtlekit.agr.TKOrganization.ENGINE_GROUP;
import static turtlekit.agr.TKOrganization.MODEL_GROUP;
import static turtlekit.agr.TKOrganization.MODEL_ROLE;
import static turtlekit.agr.TKOrganization.SCHEDULER_ROLE;
import static turtlekit.agr.TKOrganization.TURTLES_GROUP;
import static turtlekit.agr.TKOrganization.TURTLE_ROLE;
import static turtlekit.kernel.TurtleKit.Option.cuda;
import static turtlekit.kernel.TurtleKit.Option.endTime;
import static turtlekit.kernel.TurtleKit.Option.envDimension;
import static turtlekit.kernel.TurtleKit.Option.envHeight;
import static turtlekit.kernel.TurtleKit.Option.envWidth;
import static turtlekit.kernel.TurtleKit.Option.environment;
import static turtlekit.kernel.TurtleKit.Option.launcher;
import static turtlekit.kernel.TurtleKit.Option.model;
import static turtlekit.kernel.TurtleKit.Option.scheduler;
import static turtlekit.kernel.TurtleKit.Option.startSimu;
import static turtlekit.kernel.TurtleKit.Option.turtles;
import static turtlekit.kernel.TurtleKit.Option.viewers;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;

import javax.xml.parsers.ParserConfigurationException;

import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.NamedNodeMap;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;
import org.xml.sax.SAXException;

import madkit.action.SchedulingAction;
import madkit.i18n.ErrorMessages;
import madkit.kernel.Agent;
import madkit.kernel.Madkit;
import madkit.kernel.Madkit.Option;
import madkit.message.SchedulingMessage;
import madkit.util.XMLUtilities;
import turtlekit.agr.TKOrganization;
import turtlekit.cuda.CudaEngine;

public class TKLauncher extends Agent {

	
	private int width = 200;
	private int height = 200;
//	private TKScheduler sch;
	private String community;
	private Document dom;
	
	public TKLauncher() {
	}
	
	@Override
	protected void activate() {
//		setLogLevel(Level.ALL);
		initProperties();
		createGroup(TKOrganization.TK_COMMUNITY, TKOrganization.LAUNCHING);
		createGroup(community, ENGINE_GROUP);
		createGroup(community, MODEL_GROUP);
		createGroup(community, TURTLES_GROUP);
		requestRole(community, ENGINE_GROUP, MODEL_ROLE);
		if (isMadkitPropertyTrue(TurtleKit.Option.cuda) && ! CudaEngine.init()){
			setMadkitProperty(TurtleKit.Option.cuda, "false");
		}
		createSimulationInstance();
		if(isMadkitPropertyTrue(startSimu)){
			sendMessage(community, ENGINE_GROUP, SCHEDULER_ROLE, new SchedulingMessage(SchedulingAction.RUN));
		}
	}
	
	protected void live() {	}//avoid the Agent's default live message
	
	protected void createSimulationInstance(){
		if (! launchXMLAgents("Environment")) {
			launchAgent(getMadkitProperty(environment));
		}
		if (! launchXMLAgents("Scheduler")) {
//			System.err.println(getMadkitProperty(turtlekit.kernel.TurtleKit.Option.community));
			launchScheduler();
		}
		launchConfigTurtles();
		if (! launchXMLAgents("Viewer")) {
			launchViewers();
		}
		launchXMLAgents("Agent");
	}
	
	protected void launchScheduler(){
		final TKScheduler sch = (TKScheduler) launchAgent(getMadkitProperty(scheduler));
		try {
			sch.setSimulationDuration(Integer.parseInt(getMadkitProperty(endTime)));
		} catch (NumberFormatException | NullPointerException e) {
		}
	}
	
	protected void launchViewers() {
		final String viewerClasses = getMadkitProperty(viewers);
		if (viewerClasses != null && ! viewerClasses.equals("null")) {
			for (final String v : viewerClasses.split(";")) {
				launchAgent(v, true);
			}
		}
	}
	
	private NodeList getDomNodes(String nodeName){
		return dom.getElementsByTagName(nodeName);
	}
	
	private boolean launchXMLAgents(String type){
		boolean done = false;
		if (dom != null) {
			NodeList nodes = getDomNodes(type);
			for (int i = 0; i < nodes.getLength(); i++) {
				final Node item = nodes.item(i);
				if (launchNode(item) == ReturnCode.SUCCESS) {
					done = true;
				}
			}
		}
		return done;
	}
	
	private void launchXmlTurtles(){
		if (dom != null) {
			NodeList nodes = getDomNodes("Turtle");
			for (int i = 0; i < nodes.getLength(); i++) {
				final Node turtleNode = nodes.item(i);
				final NamedNodeMap namesMap = turtleNode.getAttributes();
				String[] roles = null;
				try {
					roles = namesMap.getNamedItem("roles").getNodeValue().split(";");
				} catch (NullPointerException e) {
				}
				appendBucketModeNode(turtleNode,TURTLE_ROLE);
				if (roles != null) {
					for (String role : roles) {
						appendBucketModeNode(turtleNode, role);
					}
				}
				launchNode(turtleNode);
			}
		}
	}

	/**
	 * @param turtleNode
	 * @param turtleRole 
	 */
	private void appendBucketModeNode(final Node turtleNode, String role) {
		Element turtleRole = dom.createElement(XMLUtilities.BUCKET_MODE_ROLE);
		turtleRole.setAttribute("community",community);
		turtleRole.setAttribute("group",TURTLES_GROUP);
		turtleRole.setAttribute("role", role);
		turtleNode.appendChild(turtleRole);
	}

	/**
	 * 
	 */
	protected void initProperties(){
		String modelFile = getMadkitProperty(model);
		if (modelFile != null) {
			try {
				getMadkitConfig().loadPropertiesFromFile(modelFile);
				if (modelFile.endsWith(".xml")) {
					dom = XMLUtilities.getDOM(modelFile);
				}
			} catch (SAXException | IOException | ParserConfigurationException e) {
				e.printStackTrace();
			}
		}
		community = getMadkitProperty(model);
		if (community == null) {
			community = getMadkitProperty(turtlekit.kernel.TurtleKit.Option.community);
		}
		else{
			setMadkitProperty(turtlekit.kernel.TurtleKit.Option.community,community);
		}
//		if (community == null) {
//			community = TLKOrganization.TK_COMMUNITY;
//			setMadkitProperty(communityName.name(), community);
//		}
		try {
			String widthParam = getMadkitProperty(envWidth);
			String heightParam = getMadkitProperty(envHeight);
			if (heightParam != null || widthParam != null) {
				setMadkitProperty(envDimension, (widthParam != null ? widthParam : "200") + ","
						+ (heightParam != null ? heightParam : "200"));
			}
			String[] dimension = getMadkitProperty(envDimension).split(",");
			setMadkitProperty(envWidth, dimension[0]);
			if (dimension.length > 1) {
				setMadkitProperty(envHeight, dimension[1]);
			}
			else{
				setMadkitProperty(envHeight, dimension[0]);
			}
			width = Integer.parseInt(getMadkitProperty(envWidth));
			height = Integer.parseInt(getMadkitProperty(envHeight));
		} catch (NumberFormatException e) {
			e.printStackTrace();
			setMadkitProperty(envWidth, "200");
			setMadkitProperty(envHeight, "200");
		}
	}

	/**
	 * 
	 */
	public static void main(String[] args) {
		executeThisAgent(1,false,Option.configFile.toString(),"turtlekit/kernel/turtlekit.properties",cuda.toString(), turtles.toString(),Turtle.class.getName());
	}
	

	/**
	 * This offers a convenient way to create a main method 
	 * that launches a simulation using the environment
	 * class under development. 
	 * This call only works in the main method of the environment.
	 * 
	 * @param args
	 *           MaDKit or TurtleKit options
	 * @see #executeThisAgent(int, boolean, String...)
	 * @since TurtleKit 3.0.0.1
	 */
	protected static void executeThisLauncher(String... args) {
		StackTraceElement element = null;
		for (StackTraceElement stackTraceElement : new Throwable().getStackTrace()) {
			if(stackTraceElement.getMethodName().equals("main")){
				element  = stackTraceElement;
				break;
			}
		}
		final ArrayList<String> arguments = new ArrayList<>(Arrays.asList(
				Madkit.BooleanOption.desktop.toString(),"false",
				Madkit.Option.configFile.toString(), "turtlekit/kernel/turtlekit.properties",
				launcher.toString(), element.getClassName()));
		if (args != null) {
			arguments.addAll(Arrays.asList(args));
		}
		new Madkit(arguments.toArray(new String[0]));
	}


	protected void launchConfigTurtles(){
//		Timer.startTimer(this);
//		if (logger != null)
//			logger.info("** LAUNCHING CONFIG TURTLES **");
		launchXmlTurtles();
		final String agentsTolaunch = getMadkitProperty(turtles);
		if (!agentsTolaunch.equals("null")) {
			final String[] agentsClasses = agentsTolaunch.split(";");
			for (final String classNameAndOption : agentsClasses) {
				final String[] classAndOptions = classNameAndOption.split(",");
				final String className = classAndOptions[0].trim();// TODO should test if these classes exist
				int number = 1;
				if (classAndOptions.length > 1) {
					try {
						number = Integer.parseInt(classAndOptions[1].trim());
					} catch (NumberFormatException e) {
						getLogger().severeLog(ErrorMessages.OPTION_MISUSED.toString() +Option.launchAgents.toString()+" "+ agentsTolaunch +" "+e.getClass().getName()+" !!!\n" , null);
					}
				}
				if (logger != null)
					logger.info("Launching " + number + " instance(s) of " + className);
				launchAgentBucket(className,number,Runtime.getRuntime().availableProcessors()+1,community+","+TKOrganization.TURTLES_GROUP+","+TKOrganization.TURTLE_ROLE);
			}
		}
//		Timer.stopTimer(this);
	}

	/**
	 * @return the width
	 */
	public int getWidth() {
		return width;
	}

	/**
	 * @param width the width to set
	 */
	public void setWidth(int width) {
		this.width = width;
	}

	/**
	 * @return the height
	 */
	public int getHeight() {
		return height;
	}

	/**
	 * @param height the height to set
	 */
	public void setHeight(int height) {
		this.height = height;
	}

}
