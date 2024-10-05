package junit;

import org.junit.runner.*;
import org.junit.runner.notification.*;;

public class TestRunner {
    public static void main(String[] args) {
        Result result = JUnitCore.runClasses(AllTests.class);
        for (Failure failure : result.getFailures()) {
            System.out.println(failure.getMessage());
        }
        System.out.println(result.wasSuccessful());
    }
}
