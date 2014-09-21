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
package turtlekit.viewer;

import static turtlekit.kernel.TurtleKit.Option.fastRendering;

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.Point;
import java.awt.Toolkit;
import java.awt.event.MouseWheelEvent;
import java.awt.event.MouseWheelListener;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.util.ArrayList;
import java.util.Arrays;

import javax.swing.JFrame;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JToolBar;

import madkit.action.KernelAction;
import madkit.kernel.Madkit;
import madkit.simulation.probe.SingleAgentProbe;
import madkit.simulation.viewer.SwingViewer;
import turtlekit.agr.TKOrganization;
import turtlekit.gui.CudaMenu;
import turtlekit.gui.toolbar.TKToolBar;
import turtlekit.kernel.Patch;
import turtlekit.kernel.TKEnvironment;
import turtlekit.kernel.TKScheduler;
import turtlekit.kernel.TurtleKit;

public abstract class AbstractViewer extends SwingViewer {
	
	/**
	 * 
	 */
	private static final long serialVersionUID = 671181979144051600L;
	protected int cellSize = 10;
	private SingleAgentProbe<TKEnvironment, Patch[]> gridProbe;
	private SingleAgentProbe<TKScheduler,Double> timeProbe;
	private SingleAgentProbe<TKEnvironment, Integer> widthProbe;
	private SingleAgentProbe<TKEnvironment, Integer> heightProbe;
	private String community;
	private JPanel turtlePane;
	
	protected void initProbes(){
		community = getMadkitProperty(TurtleKit.Option.community);
		gridProbe = new SingleAgentProbe<TKEnvironment, Patch[]>(
				getCommunity(), 
				TKOrganization.MODEL_GROUP, 
				TKOrganization.ENVIRONMENT_ROLE, 
				"patchGrid");
		addProbe(gridProbe);
		timeProbe = new SingleAgentProbe<>(community, TKOrganization.ENGINE_GROUP, TKOrganization.SCHEDULER_ROLE,"GVT");
		addProbe(timeProbe);
		widthProbe = new SingleAgentProbe<TKEnvironment, Integer>(
				getCommunity(), 
				TKOrganization.MODEL_GROUP, 
				TKOrganization.ENVIRONMENT_ROLE, 
				"width");
		addProbe(widthProbe);
		heightProbe = new SingleAgentProbe<TKEnvironment, Integer>(
				getCommunity(), 
				TKOrganization.MODEL_GROUP, 
				TKOrganization.ENVIRONMENT_ROLE, 
				"height");
		addProbe(heightProbe);
	}

	public AbstractViewer() {
		createGUIOnStartUp();
//		setLogLevel(Level.ALL);
	}
	
	@Override
	protected void activate() {
		setSynchronousPainting(! isMadkitPropertyTrue(fastRendering));
		requestRole(community, TKOrganization.ENGINE_GROUP,TKOrganization.VIEWER_ROLE);
	}
	
	@Override
	protected void end() {
		if(getAgentsWithRole(community, TKOrganization.ENGINE_GROUP,TKOrganization.VIEWER_ROLE) == null){
			KernelAction.EXIT.getActionFor(this).actionPerformed(null);
		}
	}
	
//	private void handleMessage(Message m){
//		if(m instanceof DimensionMessage)
//			handleDimensionMessage((DimensionMessage) m);
//	}
//
//	private void handleDimensionMessage(DimensionMessage m) {
//		Dimension d = m.getContent();
//		width = (int) d.getWidth();
//		height = (int) d.getHeight();
//		updateSize();
//	}

	@Override
	public void setupFrame(final JFrame frame) {
		initProbes();
		community = getMadkitProperty(TurtleKit.Option.community);
		SingleAgentProbe<TKScheduler,Double> p = new SingleAgentProbe<>(community, TKOrganization.ENGINE_GROUP, TKOrganization.SCHEDULER_ROLE,"GVT");
		addProbe(p);
		final TKScheduler tkScheduler = p.getCurrentAgentsList().get(0);
		final JToolBar schedulerToolBar = tkScheduler.getSchedulerToolBar();

		super.setupFrame(frame);
		
		getFrame().getJMenuBar().add(new CudaMenu(getCurrentEnvironment()));

		frame.addWindowListener(new WindowAdapter() {
			@Override
			public void windowClosing(WindowEvent e) {
				//because there may be several viewers
				if (getAgentsWithRole(getCommunity(), TKOrganization.ENGINE_GROUP, TKOrganization.VIEWER_ROLE) == null) {
					KernelAction.EXIT.getActionFor(AbstractViewer.this).actionPerformed(null);
				}
			}
		});		
		
		frame.remove(getToolBar());
		final TKToolBar comp = new TKToolBar(this);
		comp.addSeparator();
		comp.add(getToolBar());
		comp.addSeparator();
//		schedulerToolBar.setFloatable(false);
		comp.add(schedulerToolBar);
//		comp.setOrientation(JToolBar.VERTICAL);
		frame.add(comp,BorderLayout.PAGE_START);
		comp.add(tkScheduler.getGVTLabel());
		checkFrameSize();
		getDisplayPane().setBackground(Color.BLACK);
		getDisplayPane().addMouseWheelListener(new MouseWheelListener() {
			@Override
			public void mouseWheelMoved(MouseWheelEvent e) {
//				cellSize -= e.getWheelRotation();
//				if(cellSize < 1)
//					cellSize = 1;
//				receiveMessage(new DimensionMessage(width,height));
				updateSize(e.getPoint(),e.getWheelRotation());
//				Graphics2D g2 = (Graphics2D) frame.getGraphics();
//				int w = getDisplayPane().getWidth();// real width of canvas
//				int h = getDisplayPane().getHeight();// real height of canvas
//				// Translate used to make sure scale is centered
//				g2.translate(w/2, h/2);
//				g2.scale(2, 2);
//				g2.translate(-w/2, -h/2);
//				Point pt = turtlePane.getLocation();
//				turtlePane.setLocation(pt.x-e.getX(),pt.y - e.getY());
//				pt = turtlePane.getLocation();
//				System.err.println(pt);
			}

		});
		turtlePane = (JPanel) getDisplayPane();
		setDisplayPane(new JScrollPane(turtlePane));
		updateSize(null);
		frame.pack();
		frame.setLocationRelativeTo(null);
		schedulerToolBar.getComponent(0).requestFocus();
	}

	/**
	 * 
	 */
	private void checkFrameSize() {
		final Dimension screenSize = Toolkit.getDefaultToolkit().getScreenSize();
		double screenW = screenSize.getWidth();
		double screenH = screenSize.getHeight();
		while((getWidth()*cellSize > screenW - 200 || getHeight()*cellSize > screenH-300) && cellSize > 1){
			cellSize--;
		}
//		getFrame().setSize(getWidth()*cellSize, getHeight()*cellSize);
	}
	
	/**
	 * @param point
	 * @param wheelRotation
	 */
	private void updateSize(Point point, int wheelRotation) {
			final int i = point.x / cellSize;
			int offX = (int) i * wheelRotation;
			int offY = (int) (point.y / cellSize) * wheelRotation;
			cellSize -= wheelRotation;
			if (cellSize < 1)
				cellSize = 1;
			if (cellSize > 1) {
				turtlePane.setLocation(turtlePane.getLocation().x + offX,
						turtlePane.getLocation().y + offY);
			}
			else{
				turtlePane.setLocation(0, 0);
			}
			turtlePane.setPreferredSize(new Dimension(getWidth() * cellSize, getHeight() * cellSize));
//			checkFrameSize();
			turtlePane.getParent().doLayout();
			//		getFrame().setSize(
			//				new Dimension(getWidth() * cellSize + 100, getHeight()
			//						* cellSize + 100));
			//		getFrame().pack();
	}

	private void updateSize(Point p) {
		turtlePane.setPreferredSize(new Dimension(getWidth() * cellSize,
				getHeight() * cellSize));
		if (p != null) {
//			System.err.println("p" + p);
//			System.err.println("location " + turtlePane.getLocation());
			int offX = (int) (p.x * cellSize) - p.x;
			int offY = (int) (p.y * cellSize) - p.y;
//			System.err.println("offx " + offX);
//			System.err.println("offy " + offY);
//			System.err.println("new loc " + turtlePane.getLocation());
			turtlePane.setLocation(turtlePane.getLocation().x - offX,
					turtlePane.getLocation().y - offY);
		}
		else{
//			getDisplayPane().setPreferredSize(new Dimension(getWidth() * cellSize,
//					getHeight() * cellSize));
			getFrame().setSize(
					new Dimension(getWidth() * cellSize + 100, getHeight()
							* cellSize + 100));
			getFrame().pack();
		}
//		if (Toolkit.getDefaultToolkit().getScreenSize().getHeight() < getHeight()
//				* cellSize) {
//		}
//		turtlePane.getParent().doLayout();
//		System.err.println("final  loc " + turtlePane.getLocation());

	}
	
	/**
	 * @return
	 */
	public Patch[] getPatchGrid() {
		return gridProbe.getPropertyValue();
	}
	
	public Patch getPatch(int x, int y){
		return gridProbe.getPropertyValue()[x + getWidth() * y];
	}
	
	/**
	 * @return
	 */
	public int getHeight() {
		return heightProbe.getPropertyValue();
	}

	/**
	 * @return
	 */
	public int getWidth() {
		return widthProbe.getPropertyValue();
	}
	
	public TKEnvironment getCurrentEnvironment(){
		return heightProbe.getProbedAgent();
	}
	
	public double getSimulationTime(){
		return timeProbe.getPropertyValue();
	}

	/**
	 * @return the cellSize
	 */
	public int getCellSize() {
		return cellSize;
	}
	
	/**
	 * @param cellSize the cellSize to set
	 */
	public void setCellSize(int cellSize) {
		this.cellSize = cellSize;
	}

	/**
	 * @return the community
	 */
	public final String getCommunity() {
		return community;
	}
	

	/**
	 * This offers a convenient way to create a main method 
	 * that launches a simulation using the viewer
	 * class under development. 
	 * This call only works in the main method of the viewer.
	 * 
	 * @param args
	 *           MaDKit or TurtleKit options
	 * @see #executeThisAgent(int, boolean, String...)
	 * @since TurtleKit 3.0.0.1
	 */
	protected static void executeThisViewer(String... args) {
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
				TurtleKit.Option.viewers.toString(), element.getClassName()));
		if (args != null) {
			arguments.addAll(Arrays.asList(args));
		}
		new Madkit(arguments.toArray(new String[0]));
	}


}

