// A fixture for lang_class_include_test.b: a class in an INCLUDED file, with a
// factory beside it, so the test can check that both routes to an instance work
// from the includer.
class Shape {
    // The trailing underscore is not style: Bantu keeps variables and
    // functions in ONE namespace, so a field named `sides` shadows the
    // method `sides()` and calling it says so.
    def init($n)     { this.sides_ = $n; }
    def describe()   { return "Shape(" + str(this.sides_) + ")"; }
    def sides()      { return this.sides_; }
}

class Square extends Shape {
    def init() { super(4); }
    def describe() { return "Square"; }
}

// The factory pattern packages use (`nn.Linear(...)`), which has always worked
// across an include because a FUNCTION is namespaced into the module object.
def makeShape($n) { return new Shape($n); }
