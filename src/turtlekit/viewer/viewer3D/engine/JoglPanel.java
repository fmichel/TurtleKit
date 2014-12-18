package turtlekit.viewer.viewer3D.engine;

import static javax.media.opengl.fixedfunc.GLMatrixFunc.GL_MODELVIEW;
import static javax.media.opengl.fixedfunc.GLMatrixFunc.GL_PROJECTION;

import javax.media.opengl.GL2;
import javax.media.opengl.GLAutoDrawable;
import javax.media.opengl.GLCapabilities;
import javax.media.opengl.GLEventListener;
import javax.media.opengl.awt.GLJPanel;
import javax.media.opengl.glu.GLU;

import turtlekit.kernel.Patch;
import turtlekit.viewer.viewer3D.engine.tools.FPSCounter;
import turtlekit.viewer.viewer3D.engine.tools.Input;
import turtlekit.viewer.viewer3D.engine.tools.PrintText;
import turtlekit.viewer.viewer3D.engine.tools.Scene;

@SuppressWarnings("serial")
public class JoglPanel extends GLJPanel implements GLEventListener {
	
	private GLU glu;
	
	private Input input;
	private Scene scene;
	private FPSCounter fpsCounter;
	private PrintText printText;
	
	private boolean printInfoText = true;
	private boolean printFPS = true;

	private Patch[] patchGrid;	
	
	public static float rot = 0.0f;
	public static float xrot = 0.0f;
	public static float yrot = 0.0f;
	public static int lightRot;
	
	public JoglPanel(GLCapabilities capabilities, Patch[] patchs) {
		super(capabilities);
		this.addGLEventListener(this);
		this.setFocusable(true);
	    this.requestFocus();
	    patchGrid = patchs;
	}
	
	@Override
	public void init(GLAutoDrawable drawable) {
		GL2 gl = drawable.getGL().getGL2();
		
		glu = new GLU();

		gl.glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
		gl.glClearDepth(1.0f);
		gl.glEnable(GL2.GL_DEPTH_TEST);
//		gl.glDepthFunc(GL2.GL_LESS);
		gl.glDepthFunc(GL2.GL_LEQUAL);
		gl.glHint(GL2.GL_PERSPECTIVE_CORRECTION_HINT, GL2.GL_NICEST);
		gl.glShadeModel(GL2.GL_SMOOTH);
		
		gl.glEnable(GL2.GL_LINE_SMOOTH); 
		gl.glHint(GL2.GL_LINE_SMOOTH_HINT, GL2.GL_NICEST);
		gl.glLineWidth(1.0f); 

//		gl.glEnable(GL2.GL_POLYGON_SMOOTH); 
//		gl.glHint(GL2.GL_POLYGON_SMOOTH_HINT, GL2.GL_NICEST);

//		gl.glEnable(GL2.GL_POINT_SMOOTH); 
//		gl.glHint(GL2.GL_POINT_SMOOTH_HINT, GL2.GL_NICEST);

		gl.glEnable(GL2.GL_BLEND);
		gl.glBlendFunc(GL2.GL_SRC_ALPHA, GL2.GL_ONE_MINUS_SRC_ALPHA);
		
		gl.setSwapInterval(1);

		this.scene = new Scene(gl, patchGrid);
		this.input = new Input(this);
		this.printText = new PrintText(drawable, 10);
		this.fpsCounter = new FPSCounter(drawable, 10);
		
	}
	
	@Override
	public void reshape(GLAutoDrawable drawable, int x, int y, int width, int height) {
		GL2 gl = drawable.getGL().getGL2();
		 
		if (height == 0) height = 1;
		float aspect = (float)width / height;

		gl.glViewport(0, 0, width, height);
 
		gl.glMatrixMode(GL_PROJECTION);
		gl.glLoadIdentity();
		glu.gluPerspective(45.0, aspect, 0.1, 1000.0);
 
		gl.glMatrixMode(GL_MODELVIEW);
		gl.glLoadIdentity(); 
	}
	
	@Override
	public void display(GLAutoDrawable drawable) {
		scene.draw();
//		if(printFPS){
//			fpsCounter.draw();
//		}
//		if(printInfoText){
//			printText.draw(scene.getDataToPrint(), scene.getTitleToPrint());
//		}
//		scene.update();
		this.requestFocus();
	}
	
	public void displayChanged(GLAutoDrawable drawable, boolean modeChanged, boolean deviceChanged) {
		System.out.println("display changed");
		
	}	
	
	@Override
	public void dispose(GLAutoDrawable drawable) {
//		scene.dispose();
		
	}
}
