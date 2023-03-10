package linear;

public class MatrixMultiplicationTask implements Runnable {
    private final float[] a;
    private final float[] b;
    private final float[] result;
    private final int n;
    private final int startRow;
    private final int endRow;

    public MatrixMultiplicationTask(float[] a, float[] b, float[] result, int n, int startRow, int endRow) {
        this.a = a;
        this.b = b;
        this.result = result;
        this.n = n;
        this.startRow = startRow;
        this.endRow = endRow;
    }

    public void run() {
        for (int i = startRow; i < endRow; i++) {
            for (int j = 0; j < n; j++) {
                float sum = 0;

                for (int k = 0; k < n; k++) {
                    sum += a[i * n + k] * b[k * n + j];
                }

                result[i * n + j] = sum;
            }
        }
    }
}
