package linear;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.*;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaMemcpyKind;
import jcuda.runtime.dim3;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

public class LinearMultiplication {

    public LinearMultiplication() {
    }

    public void run() {
        System.out.println("-------------------------------------");
        System.out.println("|Линейный алгоритм умножения матриц |");
        System.out.println("-------------------------------------");
        testWithTime(100);
        testWithTime(500);
        testWithTime(700);
        testWithTime(1000);
        testWithTime(1200);
        testWithTime(1500);
        testWithTime(1700);
        testWithTime(2000);
        testWithTime(2200);
        testWithTime(2500);
    }

    private void testWithTime(int n) {
        float[] a = new float[n * n];
        float[] b = new float[n * n];

        for (int i = 0; i < n * n; i++) {
            a[i] = (float) Math.random();
            b[i] = (float) Math.random();
        }

        System.out.println("-----------------------------------");
        System.out.println("Размер матриц: " + n + "x" + n);

        long startTime = System.currentTimeMillis();
        multiplyMatricesLinear(a, b, n);
        long endTime = System.currentTimeMillis();
        long duration = (endTime - startTime);
        System.out.println("Линейный алгоритм: " + duration + " milliseconds");

        startTime = System.currentTimeMillis();
        multiplyMatricesWithJCUDA(a, b, n);
        endTime = System.currentTimeMillis();
        duration = (endTime - startTime);
        System.out.println("CUDA: " + duration + " milliseconds");

        startTime = System.currentTimeMillis();
        multiplyMatricesInMultiThread(a, b, n, 8);
        endTime = System.currentTimeMillis();
        duration = (endTime - startTime);
        System.out.println("MultiThreading(8 потоков): " + duration + " milliseconds");
    }

    private void multiplyMatricesLinear(float[] a, float[] b, int n) {
        float[] с = new float[n * n];

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                float sum = 0;

                for (int k = 0; k < n; k++) {
                    sum += a[i * n + k] * b[k * n + j];
                }
                с[i * n + j] = sum;
            }
        }
    }


    private void multiplyMatricesWithJCUDA(float[] a, float[] b, int n) {
        JCudaDriver.setExceptionsEnabled(true);

        float c[] = new float[n * n];

        // Allocate device memory for matrices A, B, and C
        Pointer d_A = new Pointer();
        Pointer d_B = new Pointer();
        Pointer d_C = new Pointer();
        JCuda.cudaMalloc(d_A, n * n * Sizeof.FLOAT);
        JCuda.cudaMalloc(d_B, n * n * Sizeof.FLOAT);
        JCuda.cudaMalloc(d_C, n * n * Sizeof.FLOAT);

        // Copy matrices A and B from host to device memory
        JCuda.cudaMemcpy(d_A, Pointer.to(a), n * n * Sizeof.FLOAT, cudaMemcpyKind.cudaMemcpyHostToDevice);
        JCuda.cudaMemcpy(d_B, Pointer.to(b), n * n * Sizeof.FLOAT, cudaMemcpyKind.cudaMemcpyHostToDevice);

        int threadsPerBlock = 32;
        dim3 blockSize = new dim3(threadsPerBlock, threadsPerBlock, 1);
        dim3 gridSize = new dim3((n + threadsPerBlock - 1) / threadsPerBlock, (n + threadsPerBlock - 1) / threadsPerBlock, 1);

        // Prepare the kernel function to lounch
        JCudaDriver.cuInit(0);
        CUdevice device = new CUdevice();
        JCudaDriver.cuDeviceGet(device, 0);
        CUcontext context = new CUcontext();
        JCudaDriver.cuCtxCreate(context, 0, device);

        CUmodule module = new CUmodule();
        JCudaDriver.cuModuleLoad(module, "C:\\matrix_multiply.ptx");
        CUfunction function = new CUfunction();
        JCudaDriver.cuModuleGetFunction(function, module, "matrix_multiply_kernel");

        Pointer kernelParameters = Pointer.to(
                Pointer.to(d_A),
                Pointer.to(d_B),
                Pointer.to(d_C),
                Pointer.to(new int[]{n})
        );

        // Lounch kernel function
        JCudaDriver.cuLaunchKernel(function,
                gridSize.x, gridSize.y, gridSize.z, blockSize.x, blockSize.y, blockSize.z,
                0, null, kernelParameters, null);
        JCuda.cudaDeviceSynchronize();

        // Copy matrix C from device to host memory
        JCuda.cudaMemcpy(Pointer.to(c), d_C, n * n * Sizeof.FLOAT, cudaMemcpyKind.cudaMemcpyDeviceToHost);

        // Free device memory
        JCuda.cudaFree(d_A);
        JCuda.cudaFree(d_B);
        JCuda.cudaFree(d_C);
    }


    private void multiplyMatricesInMultiThread(float[] a, float[] b, int n, int numThreads) {

        float[] result = new float[n * n];

        ExecutorService executor = Executors.newFixedThreadPool(numThreads);

        // Divide the matrix multiplication task into numThreads subtasks
        int subtaskSize = n / numThreads;

        for (int i = 0; i < numThreads; i++) {
            int startRow = i * subtaskSize;
            int endRow = (i == numThreads - 1) ? n : startRow + subtaskSize;

            executor.submit(new MatrixMultiplicationTask(a, b, result, n, startRow, endRow));
        }

        executor.shutdown();
        try {
            executor.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }

}
