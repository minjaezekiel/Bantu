// Reached only via package.json's "main". The conventional <name>.b rule would
// look for greeter/greeter.b, which does not exist -- so if this loads, the
// manifest was actually read.

def hello($who) { return "hello, " + $who; }

$ENTRY = "src/hello.b";
