package turtlekit.viewer.viewer3D.engine.tools;

import javax.media.opengl.GL2;

import turtlekit.kernel.Patch;
import turtlekit.viewer.viewer3D.engine.math.Vector3f;

/**
 * This is an example scene to demonstrate how the volume shadow creator works.
 * All the standard OpenGl stuff is done in <code>Main</code>. The only
 * additional thing that needs to be done in Main, is the following code:
 * <pre>
 * {@code
 * 		GLCapabilities capabilities = new GLCapabilities();
 *		capabilities.setStencilBits(8);
 *		GLCanvas canvas = new GLCanvas(capabilities);
 * }
 * </pre>
 *  
 * @author Tim Jörgen
 */
public class Scene {
	
	private boolean worldIn3D = false;
	
	private int DIM = 200;
	
	private Grid grid;

	private GL2 gl;
	private Camera camera;
	
//	private World world;

	/**
	 * This is our scene, that implements the ShadowScene interface. Here you do
	 * the drawing of your world and your models.
	 * @param patchGrid 
	 */

	public Scene(GL2 gl, Patch[] patchGrid) {
		this.gl = gl;
		if(worldIn3D){
			this.camera = Camera.getInstance(gl, new Vector3f(50f, 50f, 60f), new Vector3f(0f, 0f, 1f));

		}
		else{
			this.camera = Camera.getInstance(gl, new Vector3f((float)2.0f, (float)2.0f, 120f), new Vector3f(0f, 0f, 1f));
		}

//		this.world = new World(gl,DIM, worldIn3D);
//		initWorld();
		grid = new Grid(DIM, DIM, 1, patchGrid);
		grid.createGrid();
	}
	
//	public void initWorld(){
//		
//		world.addCudaToTheWorld();
//		world.addDataToTheWorld();
//		world.addGridToTheWorld(0.0f);
//
//		world.addGridToTheWorld(10.0f);
//
//		world.initWorld();
//	}

	/**
	 * This is called every frame. gl.glTranslatef(...) and gl.glRotatef(...)
	 * are done in the camera instance. Every view matrix transformation in
	 * shadow scene is done between gl.glPushMatrix and gl.glPopMatrix.
	 */
	public void draw() {
		gl.glClear(GL2.GL_COLOR_BUFFER_BIT | GL2.GL_DEPTH_BUFFER_BIT);
		gl.glLoadIdentity();

		camera.rotateAccordingToCameraPosition();
		camera.translateAccordingToCameraPosition();
		
//		gl.glColor3ub((byte)255, (byte)255, (byte)255);
//		
//		gl.glPolygonMode(GL2.GL_FRONT_AND_BACK, GL2.GL_FILL);	
//		
//		gl.glNormal3f(0,0,1);
//		
//		gl.glBegin(GL2.GL_POLYGON);
//			gl.glVertex3i(0,0, 0);
//			gl.glVertex3i(2, 0, 0);
//			gl.glVertex3i(2,2, 0);
//			gl.glVertex3i(0, 2, 0);
//		gl.glEnd();
//		world.renderWorld();
		
		grid.drawGrid(gl,false);
	}
	
//	public void update(){
//		world.updateWorld();
//	}
	
//	public void dispose(){
//		world.dispose();
//	}
	
//	public float getDataToPrint(){
//		return world.getDataToPrint();
//	}
	
//	public String getTitleToPrint(){
//		return world.getTitleToPrint();
//	}
}