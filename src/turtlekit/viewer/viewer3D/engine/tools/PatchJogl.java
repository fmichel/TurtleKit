package turtlekit.viewer.viewer3D.engine.tools;

import javax.media.opengl.GL2;

public class PatchJogl {
	
	protected int id;
	protected int size;
	
//	private Material material;

	public int getSize() {
		return size;
	}
	
	public void setSize(int size) {
		this.size = size;
	}
	
	public int getId() {
		return id;
	}
	
	public PatchJogl(int cId){
		this.id = cId;
		this.size = 1;
	}
	
	public PatchJogl(int cId, int sizePatch){
		this.id = cId;
		this.size = sizePatch;
	}
	
	public void drawPatch2D(GL2 gl, float colorData, boolean lightning){
		
//		if(lightning){
////			material = new Material(gl, colorData);
//		}
//		else{
			gl.glColor3ub((byte)colorData, (byte)colorData, (byte)colorData);
//		}
		
		gl.glPolygonMode(GL2.GL_FRONT_AND_BACK, GL2.GL_FILL);	
		
		gl.glNormal3f(0,0,1);
		
		gl.glBegin(GL2.GL_POLYGON);
			gl.glVertex3i(0,0, 0);
			gl.glVertex3i(size, 0, 0);
			gl.glVertex3i(size,size, 0);
			gl.glVertex3i(0, size, 0);
		gl.glEnd();
		
	}
	
	public void drawPatchWithEdge2D(GL2 gl, float colorData){
		
		gl.glPolygonMode(GL2.GL_FRONT_AND_BACK, GL2.GL_FILL);	
		
		gl.glColor3ub((byte)colorData, (byte)colorData, (byte)colorData);
		
		gl.glNormal3f(0,0,1);
		
		gl.glBegin(GL2.GL_POLYGON);
			gl.glVertex3i(0,0, 0);
			gl.glVertex3i(size, 0, 0);
			gl.glVertex3i(size,size, 0);
			gl.glVertex3i(0, size, 0);
		gl.glEnd();
		
		gl.glEnable(GL2.GL_POLYGON_OFFSET_LINE);
		gl.glPolygonOffset(-1, -1);
		
		gl.glPolygonMode(GL2.GL_FRONT_AND_BACK, GL2.GL_LINE);
		
		gl.glColor3f(1.0f, 1.0f, 1.0f);
		
		gl.glNormal3f(0,0,1);

		gl.glBegin(GL2.GL_POLYGON);
			gl.glVertex3i(0,0, 0);
			gl.glVertex3i(size, 0, 0);
			gl.glVertex3i(size,size, 0);
			gl.glVertex3i(0, size, 0);
		gl.glEnd();
	    
		gl.glPolygonMode(GL2.GL_FRONT_AND_BACK, GL2.GL_FILL); // Obligé de remettre en FILL lorsque l'on joue avec les polygones
		
		gl.glDisable(GL2.GL_POLYGON_OFFSET_LINE);
		
	}

	public void drawPatch3D(GL2 gl, float zData, boolean lightning){
	
		if(lightning){
//			material = new Material(gl, zData);
		}
		else{
			gl.glColor3ub((byte)zData, (byte)zData, (byte)zData);
		}
		gl.glEnable(GL2.GL_MULTISAMPLE);		
		gl.glBegin(GL2.GL_POLYGON);/* f1: front */
	        gl.glNormal3f(-size,0.0f,(25*zData)/255);
	        gl.glVertex3f(0.0f,0.0f,(25*zData)/255);
	        gl.glVertex3f(0.0f,0.0f,(25*zData)/255 + size);
	        gl.glVertex3f(size,0.0f,(25*zData)/255 + size);
	        gl.glVertex3f(size,0.0f,(25*zData)/255);
	    gl.glEnd();
	    gl.glBegin(GL2.GL_POLYGON);/* f2: bottom */
	        gl.glNormal3f(0.0f,0.0f,(25*zData)/255 - size);
	        gl.glVertex3f(0.0f,0.0f,(25*zData)/255);
	        gl.glVertex3f(size,0.0f,(25*zData)/255);
	        gl.glVertex3f(size,size,(25*zData)/255);
	        gl.glVertex3f(0.0f,size,(25*zData)/255);
        gl.glEnd();
        gl.glBegin(GL2.GL_POLYGON);/* f3:back */
	        gl.glNormal3f(size,0.0f,(25*zData)/255);
	        gl.glVertex3f(size,size,(25*zData)/255);
	        gl.glVertex3f(size,size,(25*zData)/255 + size);
	        gl.glVertex3f(0.0f,size,(25*zData)/255 + size);
	        gl.glVertex3f(0.0f,size,(25*zData)/255);
     	gl.glEnd();
     	gl.glBegin(GL2.GL_POLYGON);/* f4: top */
	        gl.glNormal3f(0.0f,0.0f,(25*zData)/255 + size);
	        gl.glVertex3f(size,size,(25*zData)/255 + size);
	        gl.glVertex3f(size,0.0f,(25*zData)/255 + size);
	        gl.glVertex3f(0.0f,0.0f,(25*zData)/255 + size);
	        gl.glVertex3f(0.0f,size,(25*zData)/255 + size);
	    gl.glEnd();
	    gl.glBegin(GL2.GL_POLYGON);/* f5: left */
	        gl.glNormal3f(0.0f,size,(25*zData)/255);
	        gl.glVertex3f(0.0f,0.0f,(25*zData)/255);
	        gl.glVertex3f(0.0f,size,(25*zData)/255);
	        gl.glVertex3f(0.0f,size,(25*zData)/255 + size);
	        gl.glVertex3f(0.0f,0.0f,(25*zData)/255 + size);
	    gl.glEnd();
	    gl.glBegin(GL2.GL_POLYGON);/* f6: right */
	        gl.glNormal3f(0.0f,-size,(25*zData)/255);
	        gl.glVertex3f(size,0.0f,(25*zData)/255);
	        gl.glVertex3f(size,0.0f,(25*zData)/255 + size);
	        gl.glVertex3f(size,size,(25*zData)/255 + size);
	        gl.glVertex3f(size,size,(25*zData)/255);
	    gl.glEnd();
	    gl.glDisable(GL2.GL_MULTISAMPLE);
	}
	
	//TODO Gerer correctement la transparence lorsque plusieurs grilles superposées!!!
	public void drawPatch3DSolid(GL2 gl, float zData, boolean lightning){
		
		if(lightning){
//			material = new Material(gl, zData);
		}
		else{
			gl.glColor4ub((byte)zData, (byte)zData, (byte)zData, (byte)zData);
		}
		
		gl.glEnable(GL2.GL_MULTISAMPLE);
		gl.glBegin(GL2.GL_POLYGON);/* f1: front */
	        gl.glNormal3f(-size,0.0f,(25*zData)/255);
	        gl.glVertex3f(0.0f,0.0f,0.0f);
	        gl.glVertex3f(0.0f,0.0f,(25*zData)/255 + size);
	        gl.glVertex3f(size,0.0f,(25*zData)/255 + size);
	        gl.glVertex3f(size,0.0f,0.0f);
	    gl.glEnd();
	    gl.glBegin(GL2.GL_POLYGON);/* f2: bottom */
	        gl.glNormal3f(0.0f,0.0f,(25*zData)/255 - size);
	        gl.glVertex3f(0.0f,0.0f,0.0f);
	        gl.glVertex3f(size,0.0f,0.0f);
	        gl.glVertex3f(size,size,0.0f);
	        gl.glVertex3f(0.0f,size,0.0f);
        gl.glEnd();
        gl.glBegin(GL2.GL_POLYGON);/* f3:back */
	        gl.glNormal3f(size,0.0f,(25*zData)/255);
	        gl.glVertex3f(size,size,0.0f);
	        gl.glVertex3f(size,size,(25*zData)/255 + size);
	        gl.glVertex3f(0.0f,size,(25*zData)/255 + size);
	        gl.glVertex3f(0.0f,size,0.0f);
     	gl.glEnd();
     	gl.glBegin(GL2.GL_POLYGON);/* f4: top */
	        gl.glNormal3f(0.0f,0.0f,(25*zData)/255 + size);
	        gl.glVertex3f(size,size,(25*zData)/255 + size);
	        gl.glVertex3f(size,0.0f,(25*zData)/255 + size);
	        gl.glVertex3f(0.0f,0.0f,(25*zData)/255 + size);
	        gl.glVertex3f(0.0f,size,(25*zData)/255 + size);
	    gl.glEnd();
	    gl.glBegin(GL2.GL_POLYGON);/* f5: left */
	        gl.glNormal3f(0.0f,size,(25*zData)/255);
	        gl.glVertex3f(0.0f,0.0f,0.0f);
	        gl.glVertex3f(0.0f,size,0.0f);
	        gl.glVertex3f(0.0f,size,(25*zData)/255 + size);
	        gl.glVertex3f(0.0f,0.0f,(25*zData)/255 + size);
	    gl.glEnd();
	    gl.glBegin(GL2.GL_POLYGON);/* f6: right */
	        gl.glNormal3f(0.0f,-size,(25*zData)/255);
	        gl.glVertex3f(size,0.0f,0.0f);
	        gl.glVertex3f(size,0.0f,(25*zData)/255 + size);
	        gl.glVertex3f(size,size,(25*zData)/255 + size);
	        gl.glVertex3f(size,size,0.0f);
	    gl.glEnd();
	    gl.glDisable(GL2.GL_MULTISAMPLE);
	}
}
