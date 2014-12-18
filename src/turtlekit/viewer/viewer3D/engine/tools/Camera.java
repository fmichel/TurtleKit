package turtlekit.viewer.viewer3D.engine.tools;

import javax.media.opengl.GL2;

import turtlekit.viewer.viewer3D.engine.math.FastMath;
import turtlekit.viewer.viewer3D.engine.math.Vector3f;

/**
 * <h2>The Camera Class</h2>
 * 
 * This initializes a simple camera on the given location. There is no framerate
 * checking done, this has to be done through an input class (or something
 * else).
 * 
 * This class is a <b>singleton</b> so you can not use the constructor. Use
 * getInstance instead.
 * 
 * To change the speed of the camera change the var called speed.
 * 
 * @author Tim Jörgen
 */
public final class Camera {

	private volatile static Camera instance;
	private GL2 gl;
	public static float xzAngle, xyAngle;
	final float speed = 0.1f;
	static Vector3f position, direction;
	static private Vector3f newPosition;
	static float newXyAngle, newXzAngle;

	/**
	 * You need to pass the arguments only once. Use null after the first frame.
	 * 
	 * @param gl
	 * @param glu
	 * @param position
	 * @param direction
	 * @return
	 */
	public static Camera getInstance(GL2 gl, Vector3f position,
			Vector3f direction) {
		if (Camera.instance == null) {
			synchronized (Camera.class) {
				Camera.instance = new Camera(gl,position, direction);
			}
		}
		return Camera.instance;
	}

	private Camera(GL2 gl, Vector3f position, Vector3f direction) {
		this.gl = gl;

		xzAngle = 0.0f;
		xyAngle = 0.0f;

		Camera.position = position;
		Camera.direction = direction;
	}

	public void rotateAccordingToCameraPosition() {
		if (Camera.xyAngle < -180.0f) {
			Camera.xyAngle = 180.0f;
		}
		if (Camera.xyAngle > 180.0f) {
			Camera.xyAngle = -180.0f;
		}
		if (Camera.xzAngle < -180.0f) {
			Camera.xzAngle = 180.0f;
		}
		if (Camera.xzAngle > 180.0f) {
			Camera.xzAngle = -180.0f;
		}
		gl.glRotatef(Camera.xyAngle, 1.0f, 0.0f, 0.0f);
		gl.glRotatef(Camera.xzAngle, 0.0f, 1.0f, 0.0f);
	}

	public void translateAccordingToCameraPosition() {
		if (Camera.cameraMoving == true) {
			smoothMoveTo();
		}
		gl.glTranslatef(-position.x, -position.y, -position.z);
	}

	public static Vector3f getPosition() {
		return position;
	}

	public void setPosition(Vector3f position) {
		Camera.position = position;
	}

	public void forward() {
		Vector3f move = new Vector3f(0.0f, 0.0f, 0.0f);
		float x, y, z;

		x = (float) Math.sin((180.0f + Camera.xzAngle) * FastMath.DEG_TO_RAD);
		y = -(float) Math.sin((180.0f + Camera.xyAngle) * FastMath.DEG_TO_RAD);
		z = -(float) Math.cos((180.0f + Camera.xzAngle) * FastMath.DEG_TO_RAD);

		move.x = x;
		move.y = y;
		move.z = z;
		move.normalize();
		move = move.mult(this.speed);

		Camera.position.subtractLocal(move);
	}

	public void backward() {
		Vector3f move = new Vector3f(0.0f, 0.0f, 0.0f);
		float x, y, z;

		x = (float) Math.sin((0.0f + Camera.xzAngle) * FastMath.DEG_TO_RAD);
		y = -(float) Math.sin((0.0f + Camera.xyAngle) * FastMath.DEG_TO_RAD);
		z = -(float) Math.cos((0.0f + Camera.xzAngle) * FastMath.DEG_TO_RAD);

		move.x = x;
		move.y = y;
		move.z = z;
		move.normalize();
		move = move.mult(this.speed);

		Camera.position.subtractLocal(move);
	}

	void up() {
		Camera.position = Camera.position.add(new Vector3f(0.0f,
				1.0f * this.speed, 0.0f));
	}

	void down() {
		Camera.position.subtractLocal(new Vector3f(0.0f,
				1.0f * this.speed, 0.0f));
	}

	void turnLeft(int dx) {
		Camera.xzAngle -= 0.5f * dx;
		updateDirection();
	}

	void turnRight(int dx) {
		Camera.xzAngle += 0.5f * dx;
		updateDirection();
	}

	void turnUp(int dy) {
		Camera.xyAngle -= 0.5f * dy;
		updateDirection();
	}

	void turnDown(int dy) {
		Camera.xyAngle += 0.5f * dy;
		updateDirection();
	}

	public void slideLeft() {
		Vector3f slide = new Vector3f(0.0f, 0.0f, 0.0f);
		float x;
		float z;

		x = (float) Math.sin((90.0f + Camera.xzAngle) * FastMath.DEG_TO_RAD);
		z = -(float) Math.cos((90.0f + Camera.xzAngle) * FastMath.DEG_TO_RAD);

		slide.x = x;
		slide.z = z;
		slide.normalize();
		slide.multLocal(this.speed);

		Camera.position.subtractLocal(slide);
	}

	public void slideRight() {
		Vector3f slide = new Vector3f(0.0f, 0.0f, 0.0f);
		float x;
		float z;

		x = (float) Math.sin((270.0f + Camera.xzAngle) * FastMath.DEG_TO_RAD);
		z = -(float) Math.cos((270.0f + Camera.xzAngle) * FastMath.DEG_TO_RAD);

		slide.x = x;
		slide.z = z;
		slide.normalize();
		slide.multLocal(this.speed);

		Camera.position.subtractLocal(slide);
	}

	private void updateDirection() {
		float x;
		float y = 0.0f;
		float z;
		x = -(float) Math.sin(degreeToRadian(Camera.xzAngle));
		y = (float) Math.sin(degreeToRadian(Camera.xyAngle));
		z = (float) Math.cos(degreeToRadian(Camera.xzAngle));

		Camera.direction.x = x;
		Camera.direction.y = y;
		Camera.direction.z = z;
	}

	double degreeToRadian(float angle) {
		return angle * Math.PI / 180.0f;
	}

	static boolean cameraMoving = false;

	public static void smoothMoveTo() {
		if (Math.abs(Camera.position.x - newPosition.x) < 33.0f
				&& Math.abs(Camera.position.y - newPosition.y) < 33.0f
				&& Math.abs(Camera.position.z - newPosition.z) < 33.0f
				&& Math.abs(Camera.xyAngle - newXyAngle) < 5.0f
				&& Math.abs(Camera.xzAngle - newXzAngle) < 5.0f) {
			cameraMoving = false;
			return;
		}

		float xFactor = -1;
		float yFactor = -1;
		float zFactor = -1;
		if ((Camera.position.x - newPosition.x) < 0) {
			xFactor = 1;
		}
		if ((Camera.position.y - newPosition.y) < 0) {
			yFactor = 1;
		}
		if ((Camera.position.z - newPosition.z) < 0) {
			zFactor = 1;
		}
		Camera.position.x = Camera.position.x + 16.0f * xFactor;
		Camera.position.y = Camera.position.y + 16.0f * yFactor;
		Camera.position.z = Camera.position.z + 16.0f * zFactor;

		float xyAngleFactor = -1;
		if ((Math.abs(newXyAngle - Camera.xyAngle) > 2.0f)) {
			if (Camera.xyAngle - newXyAngle > 0) {
				xyAngleFactor = 1;
			}
			Camera.xyAngle -= 1.0f * xyAngleFactor;
		}

		float xzAngleFactor = -1;
		if ((Math.abs(newXzAngle - Camera.xzAngle) > 2.0f)) {
			if (Camera.xzAngle - newXzAngle > 0) {
				xzAngleFactor = 1;
			}
			Camera.xzAngle -= (1.0f * xzAngleFactor);
		}
	}

	public static void smoothMoveTo(Vector3f newPosition, float xyAngle,
			float xzAngle) {
		Camera.newPosition = newPosition;
		Camera.newXyAngle = xyAngle;
		Camera.newXzAngle = xzAngle;
		Camera.cameraMoving = true;
	}
}
