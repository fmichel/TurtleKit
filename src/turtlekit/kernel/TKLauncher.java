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
import java.math.BigDecimal;
import java.time.temporal.ChronoUnit;
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
import madkit.kernel.Scheduler;
import madkit.message.SchedulingMessage;
import madkit.util.XMLUtilities;
import turtlekit.agr.TKOrganization;
import turtlekit.cuda.CudaEngine;
import turtlekit.kernel.TurtleKit.LevelOption;

public class TKLauncher extends Agent {

    private String community;

    private int width = 200;
    private int height = 200;
    // private TKScheduler sch;
    /**
     * @return the community
     */
    public String getCommunity() {
	return community;
    }

    private Document dom;
    private TKSimulationEngine simulationModel;

    public TKLauncher() {
    }

    private TKSimulationEngine simulationEngine;

    protected void initSimulationMainProperties() {
	String modelFile = getMadkitProperty(model);
	if (modelFile != null) {
	    try {
		getMadkitConfig().loadPropertiesFromFile(modelFile);
		if (modelFile.endsWith(".xml")) {
		    dom = XMLUtilities.getDOM(modelFile);
		}
	    }
	    catch(SAXException | IOException | ParserConfigurationException e) {
		e.printStackTrace();
	    }
	}
	community = getMadkitProperty(model);
	if (community == null) {
	    community = getMadkitProperty(turtlekit.kernel.TurtleKit.Option.community);
	}
	else {
	    setMadkitProperty(turtlekit.kernel.TurtleKit.Option.community, community);
	}
	String widthParam = getMadkitProperty(envWidth);
	String heightParam = getMadkitProperty(envHeight);
	if (heightParam != null || widthParam != null) {
	    setMadkitProperty(envDimension, (widthParam != null ? widthParam : "200") + "," + (heightParam != null ? heightParam : "200"));
	}
	String[] dimension = getMadkitProperty(envDimension).split(",");
	setMadkitProperty(envWidth, dimension[0]);
	if (dimension.length > 1) {
	    setMadkitProperty(envHeight, dimension[1]);
	}
	else {
	    setMadkitProperty(envHeight, dimension[0]);
	}
    }

    protected void launchViewers() {
	final String viewerClasses = getMadkitProperty(viewers);
	if (viewerClasses != null && !viewerClasses.equals("null")) {
	    for (final String v : viewerClasses.split(";")) {
		    launchAgent(v, true);
	    }
	}
    }

    protected void launchScheduler() {
	if (!launchXMLAgents("Scheduler")) {
	    final Scheduler sch = (Scheduler) launchAgent(getMadkitProperty(scheduler));
	    try {
		sch.getSimulationTime().setEndTick(BigDecimal.valueOf(Integer.parseInt(getMadkitProperty(endTime))));
	    }
	    catch(UnsupportedOperationException e) {
		sch.getSimulationTime().setEndDate(sch.getSimulationTime().getCurrentDate().plus(Integer.parseInt(getMadkitProperty(endTime)), ChronoUnit.DAYS));
	    }
	    catch(NumberFormatException | NullPointerException e) {
	    }
	}
    }

    /**
     *
     */
    protected void launchCoreAgents() {
	launchScheduler();
	if (!launchXMLAgents("Environment")) {
	    launchAgent(getMadkitProperty(environment));
	}
	simulationEngine = new TKSimulationEngine(this);
	launchAgent(simulationEngine);
    }

    /**
     * Creates the simulation agents.
     *
     * 1. launch core agents : Environment -> Scheduler -> Simulation model 2. launchConfigTurtles : launchXmlTurtles ->
     * launch args turtles 3. launch viewers 4. launch xml Agent tag
     */
    protected void createSimulationInstance() {
	launchCoreAgents();
	launchConfigTurtles();
	if (!launchXMLAgents("Viewer")) {
//	    startFX();
	    launchViewers();
	}
	launchXMLAgents("Agent");
    }

    @Override
    protected void activate() {
	initSimulationMainProperties();
	createGroup(community, ENGINE_GROUP);
	requestRole(community, ENGINE_GROUP, TKOrganization.LAUNCHER_ROLE);
	createGroup(community, MODEL_GROUP);
	createGroup(community, TURTLES_GROUP);
	requestRole(community, ENGINE_GROUP, MODEL_ROLE);// FIXME

	if (isMadkitPropertyTrue(TurtleKit.Option.cuda) && !CudaEngine.init(getMadkitProperty(LevelOption.turtleKitLogLevel))) {
	    setMadkitProperty(TurtleKit.Option.cuda, "false");
	}
	createSimulationInstance();
	if (isMadkitPropertyTrue(startSimu)) {
	    sendMessage(community, ENGINE_GROUP, SCHEDULER_ROLE, new SchedulingMessage(SchedulingAction.RUN));
	}
    }

    protected void live() {
    }// avoid the Agent's default live message

    private NodeList getDomNodes(String nodeName) {
	return dom.getElementsByTagName(nodeName);
    }

    private boolean launchXMLAgents(String type) {
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

    private void launchXmlTurtles() {
	if (dom != null) {
	    NodeList nodes = getDomNodes("Turtle");
	    for (int i = 0; i < nodes.getLength(); i++) {
		final Node turtleNode = nodes.item(i);
		final NamedNodeMap namesMap = turtleNode.getAttributes();
		String[] roles = null;
		try {
		    roles = namesMap.getNamedItem("roles").getNodeValue().split(";");
		}
		catch(NullPointerException e) {
		}
		appendBucketModeNode(turtleNode, TURTLE_ROLE);
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
	turtleRole.setAttribute("community", community);
	turtleRole.setAttribute("group", TURTLES_GROUP);
	turtleRole.setAttribute("role", role);
	turtleNode.appendChild(turtleRole);
    }

    /**
     *
     */
    public static void main(String[] args) {
	executeThisAgent(1, false, Option.configFile.toString(), "turtlekit/kernel/turtlekit.properties", cuda.toString(), turtles.toString(), Turtle.class.getName());
    }

    /**
     * This offers a convenient way to create a main method that launches a simulation using the environment class under
     * development. This call only works within a classic main static method.
     *
     * @param args
     *            MaDKit or TurtleKit options to add
     * @see #executeThisAgent(int, boolean, String...)
     * @since TurtleKit 3.0.0.1
     */
    protected static void executeThisLauncher(String... args) {
	StackTraceElement element = null;
	for (StackTraceElement stackTraceElement : new Throwable().getStackTrace()) {
	    if (stackTraceElement.getMethodName().equals("main")) {
		element = stackTraceElement;
		break;
	    }
	}
	final ArrayList<String> arguments = new ArrayList<>(Arrays.asList(Madkit.BooleanOption.desktop.toString(), "false", Madkit.Option.configFile.toString(),
		"turtlekit/kernel/turtlekit.properties", launcher.toString(), element.getClassName()));
	if (args != null) {
	    arguments.addAll(Arrays.asList(args));
	}
	new Madkit(arguments.toArray(new String[0]));
    }

    protected void launchConfigTurtles() {
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
		    }
		    catch(NumberFormatException e) {
			getLogger().severeLog(
				ErrorMessages.OPTION_MISUSED.toString() + Option.launchAgents.toString() + " " + agentsTolaunch + " " + e.getClass().getName() + " !!!\n", null);
		    }
		}
		launchAgentBucket(className, number, Runtime.getRuntime().availableProcessors() + 1,
			community + "," + TKOrganization.TURTLES_GROUP + "," + TKOrganization.TURTLE_ROLE);
	    }
	}
    }

    /**
     * @return the simulationModel
     */
    public TKSimulationEngine getSimulationEngine() {
	return simulationEngine;
    }

}
