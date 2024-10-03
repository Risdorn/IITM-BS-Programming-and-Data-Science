package junit;
import static org.junit.Assert.*;
import org.junit.Test;
public class junittrial {
    @Test
    public void testAdd() {
        Calculator calculator = new Calculator(10, 20);
        assertEquals(30, calculator.add());
    }
    @Test
    public void testMultiply() {
        Calculator calculator = new Calculator(10, 20);
        assertEquals(200, calculator.multiply());
    }
}
