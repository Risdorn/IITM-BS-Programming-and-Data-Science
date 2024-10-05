package junit;

import static org.junit.Assert.*;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;

public class JUnitTest2 {
    private int n1;
    private int n2;
    private Calculator c;

    @Before
    public void setUp() {
        n1 = 10;
        n2 = 20;
        c = new Calculator(n1, n2);
    }

    @Test
    public void testMultiply() {
        assertEquals(200, c.multiply());
    }

    @After
    public void tearDown() {
        
    }
}
