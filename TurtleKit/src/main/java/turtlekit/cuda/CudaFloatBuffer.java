package turtlekit.cuda;

import java.nio.FloatBuffer;

public class CudaFloatBuffer extends CudaUnifiedBuffer {

	FloatBuffer values;

	public CudaFloatBuffer(CudaObject co) {
		super(co);
		values = co.getCudaEngine().getUnifiedBufferBetweenPointer(getPinnedMemory(), getDevicePtr(),
				Float.class, co.getWidth(), co.getHeight());
	}

	public Float get(int index) {
		return values.get(index);
	}

	public void put(int index, Float value) {
		values.put(index, value);
	}

	public FloatBuffer getValues() {
		return values;
	}

	public int size() {
		return values.capacity();
	}

}
