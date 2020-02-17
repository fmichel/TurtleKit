package turtlekit.cuda;

import jcuda.Pointer;
import jcuda.driver.CUdeviceptr;

public abstract class CudaUnifiedBuffer {

    private Pointer pinnedMemory;
    private CUdeviceptr devicePtr;
    private Pointer dataPpointer;
    private CudaObject cudaObject;

    public CudaUnifiedBuffer(CudaObject co) {
	this.cudaObject = co;
	pinnedMemory = new Pointer();
	devicePtr = new CUdeviceptr();
	dataPpointer = Pointer.to(pinnedMemory);
    }
    
    public Pointer getPointer() {
	return Pointer.to(pinnedMemory);
    }
    
    public void freeMemory() {
	cudaObject.getCudaEngine().freeCudaMemory(pinnedMemory);
	cudaObject.getCudaEngine().freeCudaMemory(devicePtr);
    }

    
    /**
     * @return the cudaObject
     */
    public CudaObject getCudaObject() {
        return cudaObject;
    }

    
    /**
     * @return the pinnedMemory
     */
    Pointer getPinnedMemory() {
        return pinnedMemory;
    }

    
    /**
     * @return the devicePtr
     */
    CUdeviceptr getDevicePtr() {
        return devicePtr;
    }

    
    /**
     * @return the dataPpointer
     */
    Pointer getDataPpointer() {
        return dataPpointer;
    }


}
