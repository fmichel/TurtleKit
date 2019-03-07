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
package turtlekit.cuda;

import java.nio.Buffer;

import jcuda.Pointer;
import jcuda.driver.CUdeviceptr;

public interface CudaObject {
    
	public void freeMemory();
	
	public int getWidth();
	
	public int getHeight();
	
	default CudaEngine getCudaEngine(){
		return CudaEngine.getCudaEngine(this);
	}
	
	default KernelConfiguration getNewKernelConfiguration(){
		return getCudaEngine().getDefaultKernelConfiguration(getWidth(), getHeight());
	}
	default CudaKernel getCudaKernel(final String kernelFunctionName, final String cuSourceFilePath, final KernelConfiguration kc){
		return getCudaEngine().getKernel(kernelFunctionName, cuSourceFilePath, kc);
	}
	
	default Pointer getPointerToFloat(float f){
		return Pointer.to(new float[]{f});
	}
	
	default Pointer getPointerToInt(int i){
		return Pointer.to(new int[]{i});
	}
	
	default <T> CUdeviceptr createDeviceDataGrid(Class<T> dataType){
		return getCudaEngine().createDeviceDataGrid(getWidth(), getHeight(), dataType);
	}
	
	default <T> Buffer getUnifiedBufferBetweenPointer(Pointer hostData, CUdeviceptr deviceData, Class<T> dataType){
		return getCudaEngine().getUnifiedBufferBetweenPointer(hostData, deviceData, dataType, getWidth(), getHeight());
	}
	
	default void freeCudaMemory(Pointer p){
		getCudaEngine().freeCudaMemory(p);
	}
	

	

}
