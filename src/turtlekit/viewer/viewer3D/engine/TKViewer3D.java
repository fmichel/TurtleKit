package turtlekit.viewer.viewer3D.engine;

import java.awt.BorderLayout;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.Toolkit;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;

import javax.media.opengl.GLCapabilities;
import javax.media.opengl.GLProfile;
import javax.swing.JFrame;
import javax.swing.JScrollPane;
import javax.swing.JToolBar;

import madkit.action.KernelAction;
import madkit.kernel.Madkit.BooleanOption;
import madkit.simulation.probe.SingleAgentProbe;
import madkit.simulation.viewer.SwingViewer;
import turtlekit.agr.TKOrganization;
import turtlekit.gui.toolbar.TKToolBar;
import turtlekit.kernel.Patch;
import turtlekit.kernel.TKEnvironment;
import turtlekit.kernel.TKScheduler;
import turtlekit.kernel.Turtle;
import turtlekit.kernel.TurtleKit;
import turtlekit.kernel.TurtleKit.Option;

import com.jogamp.opengl.util.FPSAnimator;

public class TKViewer3D extends SwingViewer {
	
	protected int cellSize = 10;
	private SingleAgentProbe<TKEnvironment, Patch[]> gridProbe;
	private SingleAgentProbe<TKScheduler,Double> timeProbe;
	private SingleAgentProbe<TKEnvironment, Integer> widthProbe;
	private SingleAgentProbe<TKEnvironment, Integer> heightProbe;
	
	private JoglPanel joglPanel;
	private static FPSAnimator animator;
	private static int numberFPS = 120;
	
	public static FPSAnimator getAnimator() {
		return animator;
	}

	//FIXME Refactor
	protected void initProbes(){
		gridProbe = new SingleAgentProbe<TKEnvironment, Patch[]>(
				getCommunity(), 
				TKOrganization.MODEL_GROUP, 
				TKOrganization.ENVIRONMENT_ROLE, 
				"patchGrid");
		addProbe(gridProbe);
		timeProbe = new SingleAgentProbe<>(getCommunity(), TKOrganization.ENGINE_GROUP, TKOrganization.SCHEDULER_ROLE,"GVT");
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
	
	/**
	 * shortcut for getMadkitProperty(TurtleKit.Option.community)
	 * 
	 * @return the community of the simulation this agent is in
	 */
	//FIXME Refactor
	public final String getCommunity() {
		return getMadkitProperty(TurtleKit.Option.community);
	}
	
	@Override
	protected void activate() {
		requestRole(getCommunity(), TKOrganization.ENGINE_GROUP, TKOrganization.VIEWER_ROLE);
		animator.start();
	}
	
	public TKViewer3D() {
		createGUIOnStartUp();
	}
	
	@Override
	public void setupFrame(final JFrame frame) {
		initProbes();
		SingleAgentProbe<TKScheduler,Double> p = new SingleAgentProbe<>(getCommunity(), TKOrganization.ENGINE_GROUP, TKOrganization.SCHEDULER_ROLE,"GVT");
		addProbe(p);
		final TKScheduler tkScheduler = p.getCurrentAgentsList().get(0);
		final JToolBar schedulerToolBar = tkScheduler.getSchedulerToolBar();

		super.setupFrame(frame);
		
//		getFrame().getJMenuBar().add(new CudaMenu(getCurrentEnvironment()));

		frame.addWindowListener(new WindowAdapter() {
			@Override
			public void windowClosing(WindowEvent e) {
				//because there may be several viewers
				if (getAgentsWithRole(getCommunity(), TKOrganization.ENGINE_GROUP, TKOrganization.VIEWER_ROLE) == null) {
					KernelAction.EXIT.getActionFor(TKViewer3D.this).actionPerformed(null);
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
		
		GLProfile profile = GLProfile.get(GLProfile.GL2);
		GLCapabilities capabilities = new GLCapabilities(profile);
		capabilities.setNumSamples(2);
		
		initMainPanel(capabilities);
		
		setDisplayPane(new JScrollPane(joglPanel));
	    
		//Ajout du WindowListener sur la JFrame
	    frame.addWindowListener(new WindowAdapter() {
	    	@Override
	    	public void windowClosing(WindowEvent e) {
	    		// Use a dedicate thread to run the stop() to ensure that the
	    		// animator stops before program exits.
	    		new Thread() {
	    			@Override
	    			public void run() {
	    				if (animator.isStarted()) animator.stop();
	    				System.exit(0);
	    			}
	    		}.start();
	    	}
	    });
//	    animator.start(); 	//Démarrage de l'animation

		frame.pack();
		frame.setLocationRelativeTo(null);
		schedulerToolBar.getComponent(0).requestFocus();
	}
	
	public void initMainPanel(GLCapabilities capabilities){
		joglPanel = new JoglPanel(capabilities, getPatchGrid());
		joglPanel.setPreferredSize(new Dimension(getWidth()*cellSize, getHeight()*cellSize));
		
		animator = new FPSAnimator(joglPanel, numberFPS, true); 
	}

	//FIXME
	private void checkFrameSize() {
		final Dimension screenSize = Toolkit.getDefaultToolkit().getScreenSize();
		double screenW = screenSize.getWidth();
		double screenH = screenSize.getHeight();
		while((getWidth()*cellSize > screenW - 200 || getHeight()*cellSize > screenH-300) && cellSize > 1){
			cellSize--;
		}
	//	getFrame().setSize(getWidth()*cellSize, getHeight()*cellSize);
	}
	
	/**
	 * @return
	 */
	//FIXME
	public int getHeight() {
		return heightProbe.getPropertyValue();
	}

	/**
	 * @return
	 */
	//FIXME
	public int getWidth() {
		return widthProbe.getPropertyValue();
	}
	
	@Override
	protected void render(Graphics arg0) {
		// TODO Auto-generated method stub
		
	}
	
	/**
	 * @return
	 */
	public Patch[] getPatchGrid() {
		return gridProbe.getPropertyValue();
	}
	
	public static void main(String[] args) {
		new TurtleKit("--turtles", Turtle.class.getName()+",100", Option.viewers.toString(), TKViewer3D.class.getName(), BooleanOption.desktop.toString(), "false");
	}

}
