package turtlekit.cuda;

import jcuda.driver.CUstream;

public class KernelConfiguration {
	
	private int gridDimX;
	private int gridDimY;
	private int blockDimX;
	private int blockDimY;
	private CUstream streamID;
	
	
	public KernelConfiguration(int gridDimX, int gridDimY, int blockDimX, int blockDimY, CUstream streamID) {
		super();
		this.gridDimX = gridDimX;
		this.gridDimY = gridDimY;
		this.blockDimX = blockDimX;
		this.blockDimY = blockDimY;
		this.streamID = streamID;
	}


	public int getGridDimX() {
		return gridDimX;
	}


	public void setGridDimX(int gridDimX) {
		this.gridDimX = gridDimX;
	}


	public int getGridDimY() {
		return gridDimY;
	}


	public void setGridDimY(int gridDimY) {
		this.gridDimY = gridDimY;
	}


	public int getBlockDimX() {
		return blockDimX;
	}


	public void setBlockDimX(int blockDimX) {
		this.blockDimX = blockDimX;
	}


	public int getBlockDimY() {
		return blockDimY;
	}


	public void setBlockDimY(int blockDimY) {
		this.blockDimY = blockDimY;
	}


	public CUstream getStreamID() {
		return streamID;
	}


	public void setStreamID(CUstream streamID) {
		this.streamID = streamID;
	}
	
	
}
