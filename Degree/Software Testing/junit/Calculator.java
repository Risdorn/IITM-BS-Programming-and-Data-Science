package junit;

public class Calculator {
    private final int a;
    private final int b;

    public Calculator(int a, int b){
        this.a = a;
        this.b = b;
    }

    public int add(){
        return this.a + this.b;
    }

    public int multiply(){
        return this.a * this.b;
    }
}
