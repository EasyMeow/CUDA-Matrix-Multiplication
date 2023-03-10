#define BLOCK_SIZE 32

__global__ void matrixMultiplicationKernel(float* a, float* b, float* c, int n) {
    __shared__ float sharedA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float sharedB[BLOCK_SIZE][BLOCK_SIZE];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0;

    for (int k = 0; k < n; k += BLOCK_SIZE) {
        sharedA[threadIdx.y][threadIdx.x] = a[row * n + k + threadIdx.x];
        sharedB[threadIdx.y][threadIdx.x] = b[(k + threadIdx.y) * n + col];

        __syncthreads();

        for (int i = 0; i < BLOCK_SIZE; i++) {
            sum += sharedA[threadIdx.y][i] * sharedB[i][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < n && col < n) {
        c[row * n + col] = sum;
    }
}
