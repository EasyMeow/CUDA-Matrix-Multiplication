package block;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.*;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaMemcpyKind;
import jcuda.runtime.dim3;

public class BlockMultiplication {

    public BlockMultiplication() {
    }

    public void run() throws InterruptedException {
        System.out.println("------------------------------------");
        System.out.println("|Блочный алгоритм умножения матриц |");
        System.out.println("------------------------------------");
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

    private void testWithTime(int n) throws InterruptedException {
        float[] a = new float[n * n];
        float[] b = new float[n * n];

        for (int i = 0; i < n * n; i++) {
            a[i] = (float) Math.random();
            b[i] = (float) Math.random();
        }

        System.out.println("-----------------------------------");
        System.out.println("Размер матриц: " + n + "x" + n);

        long startTime = System.currentTimeMillis();
        blockMatrixMultiplication(a, b, n, n/10);
        long endTime = System.currentTimeMillis();
        long duration = (endTime - startTime);
        System.out.println("Блочный алгоритм: " + duration + " milliseconds");

        startTime = System.currentTimeMillis();
        blockMatrixMultiplicationCUDA(a, b, n, n/10);
        endTime = System.currentTimeMillis();
        duration = (endTime - startTime);
        System.out.println("CUDA: " + duration + " milliseconds");

        startTime = System.currentTimeMillis();
        matrixMultiplyBlockMultiThread(a, b, n, n/10, 8);
        endTime = System.currentTimeMillis();
        duration = (endTime - startTime);
        System.out.println("MultiThreading(8 потоков): " + duration + " milliseconds");

    }

    private void blockMatrixMultiplication(float[] a, float[] b, int n, int blockSize) {
        float[] c = new float[n * n];

        for (int i = 0; i < n; i += blockSize) {
            for (int j = 0; j < n; j += blockSize) {
                for (int k = 0; k < n; k += blockSize) {
                    for (int l = i; l < Math.min(i + blockSize, n); l++) {
                        for (int m = j; m < Math.min(j + blockSize, n); m++) {
                            float sum = 0;

                            for (int p = k; p < Math.min(k + blockSize, n); p++) {
                                sum += a[l * n + p] * b[p * n + m];
                            }

                            c[l * n + m] += sum;
                        }
                    }
                }
            }
        }
    }


    private void blockMatrixMultiplicationCUDA(float[] a, float[] b, int n, int block) {
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
        Pointer kernelParams = Pointer.to(
                Pointer.to(d_A),
                Pointer.to(d_B),
                Pointer.to(d_C),
                Pointer.to(new int[] { n }),
                Pointer.to(new int[] { block })
        );

        // Lounch kernel function
        JCudaDriver.cuLaunchKernel(function,
                gridSize.x, gridSize.y, gridSize.z, blockSize.x, blockSize.y, blockSize.z,
                0, null, kernelParams, null);
        JCuda.cudaDeviceSynchronize();

        // Copy matrix C from device to host memory
        JCuda.cudaMemcpy(Pointer.to(c), d_C, n * n * Sizeof.FLOAT, cudaMemcpyKind.cudaMemcpyDeviceToHost);

        // Free device memory
        JCuda.cudaFree(d_A);
        JCuda.cudaFree(d_B);
        JCuda.cudaFree(d_C);
    }

    private void matrixMultiplyBlockMultiThread(float[] a, float[] b, int n, int blockSize, int numThreads) throws InterruptedException {
        float[] c = new float[n * n];

        Thread[] threads = new Thread[numThreads];
        final int blocksPerThread = (int) Math.ceil(n / (double) (blockSize * numThreads));

        for (int t = 0; t < numThreads; t++) {
            final int startBlock = t * blocksPerThread;
            final int endBlock = Math.min((t + 1) * blocksPerThread, n / blockSize);

            threads[t] = new Thread(() -> {
                for (int iBlock = startBlock; iBlock < endBlock; iBlock++) {
                    for (int jBlock = 0; jBlock < n / blockSize; jBlock++) {
                        for (int kBlock = 0; kBlock < n / blockSize; kBlock++) {
                            for (int i = iBlock * blockSize; i < (iBlock + 1) * blockSize && i < n; i++) {
                                for (int j = jBlock * blockSize; j < (jBlock + 1) * blockSize && j < n; j++) {
                                    float sum = 0.0f;
                                    for (int k = kBlock * blockSize; k < (kBlock + 1) * blockSize && k < n; k++) {
                                        sum += a[i * n + k] * b[k * n + j];
                                    }
                                    c[i * n + j] += sum;
                                }
                            }
                        }
                    }
                }
            });
            threads[t].start();
        }

        for (int t = 0; t < numThreads; t++) {
            threads[t].join();
        }
    }

}
