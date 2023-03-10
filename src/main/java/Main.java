import block.BlockMultiplication;
import linear.LinearMultiplication;

public class Main {

    public static void main(String[] args) throws InterruptedException {

        LinearMultiplication linearMultiplication = new LinearMultiplication();
        linearMultiplication.run();
        BlockMultiplication blockMultiplication = new BlockMultiplication();
        blockMultiplication.run();
    }

}
