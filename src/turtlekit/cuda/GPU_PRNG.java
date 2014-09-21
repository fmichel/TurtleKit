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

import static jcuda.runtime.JCuda.cudaFree;
import static jcuda.runtime.JCuda.cudaMalloc;
import static jcuda.runtime.JCuda.cudaMemcpy;

import java.nio.BufferUnderflowException;
import java.nio.FloatBuffer;
import java.util.Random;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.jcurand.JCurand;
import jcuda.jcurand.curandGenerator;
import jcuda.jcurand.curandRngType;
import jcuda.runtime.cudaMemcpyKind;

public class GPU_PRNG extends Random implements CudaObject{

	private FloatBuffer fb;

	public GPU_PRNG() {
	}

	public GPU_PRNG(long seed) {
		super(seed);
//		CudaEngine cuda = CudaEngine.getCudaEngine(this);
		// Allocate device memory 
		int n = 10_000_000;
		Pointer deviceData = new Pointer();
		cudaMalloc(deviceData, n * Sizeof.FLOAT);
		
		// Create and initialize a pseudo-random number generator 
		curandGenerator generator = new curandGenerator();
		JCurand.curandCreateGenerator(generator,curandRngType.CURAND_RNG_PSEUDO_DEFAULT);
		JCurand.curandSetPseudoRandomGeneratorSeed(generator, seed);
		
		// Generate random numbers 
		JCurand.curandGenerateUniform(generator, deviceData, n);
		
		// Copy the random numbers from the device to the host 
		float[] hostData = new float[n];
		fb = FloatBuffer.wrap(hostData);
//		fb.compact();
		fb.rewind();
		System.err.println("\n-------------------"+fb.isDirect());
		cudaMemcpy(Pointer.to(hostData), deviceData, n * Sizeof.FLOAT, cudaMemcpyKind.cudaMemcpyDeviceToHost); 
        // Cleanup 
		JCurand.curandDestroyGenerator(generator);
		cudaFree(deviceData);
	}
	
	@Override
	public float nextFloat() {
//		try {
//			return hostData[index++];
//		} catch (ArrayIndexOutOfBoundsException e) {
//			index=0;
//		}
//		return nextFloat();
		try {
			return fb.get();
		} catch (BufferUnderflowException e) {
			fb.rewind();
		}
		return nextFloat();
	}
	
	@Override
	public double nextDouble() {
		return nextFloat();
	}

	@Override
	public void freeMemory() {
		
	}

}
