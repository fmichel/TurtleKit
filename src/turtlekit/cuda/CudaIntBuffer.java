package turtlekit.cuda;

import java.nio.IntBuffer;

public class CudaIntBuffer extends CudaUnifiedBuffer {

    IntBuffer values;

    public CudaIntBuffer(CudaObject co) {
	super(co);
	values = (IntBuffer) co.getCudaEngine().getUnifiedBufferBetweenPointer(getPinnedMemory(), getDevicePtr(), Integer.class, co.getWidth(), co.getHeight());
    }

    public int get(int index) {
	return values.get(index);
    }
    
    public void put(int index, Integer value) {
	values.put(index, value);
    }

}
