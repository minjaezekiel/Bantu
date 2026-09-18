#pragma once
/**
 * Bantu Language - Abstract Syntax Tree Nodes
 */

#include "types.hpp"
#include <vector>
#include <memory>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <typeinfo>

// Every node carries a tag naming its own type.
//
// WHY: Evaluator::evalNode used to dispatch by trying dynamic_cast against each
// node type in turn -- 38 of them, ordered by when each was added rather than by
// how often it runs, so a CallNode paid for nineteen failed casts before
// reaching its own. A failing dynamic_cast is not a comparison; it calls into
// libc++abi and walks the class hierarchy comparing type_info records. Profiled
// on a 20M-iteration arithmetic loop, that dispatcher was **79.6% of all
// interpreter time** -- four times everything else in the process put together,
// including Value's size, the allocator and the environment's string hashing.
// See docs/interpreter-performance.md for the full profile.
//
// The tag turns dispatch into one byte load and one jump table: O(1), and the
// same cost for the 39th node type as for the 1st.
//
// SAFETY: a tag plus static_cast is fast precisely because it skips the runtime
// check, so a tag that disagreed with its type would be undefined behaviour.
// It cannot disagree, because no node sets its own tag by hand -- each derives
// from ASTNodeK<K>, which passes K up, and NODE_AS<T> compares against
// T::kKind, read back off the same type. The mapping is structural, not
// clerical. Belt and braces on top: -DBANTU_CHECK_NODEKIND turns every cast
// into a checked one (the whole test suite is run that way as a phase gate),
// and evalNode keeps the old dynamic_cast chain as its default: arm, so an
// unmapped node evaluates slowly rather than wrongly.
enum class NodeKind : uint8_t {
    Number,
    String,
    Bool,
    Null,
    List,
    Dict,
    Variable,
    VarDecl,
    Assign,
    BinaryOp,
    UnaryOp,
    If,
    While,
    For,
    Each,
    Break,
    Continue,
    Throw,
    Switch,
    FuncDecl,
    Return,
    Call,
    DotAccess,
    IndexAccess,
    IndexAssign,
    DictAssign,
    TryCatch,
    ClassDecl,
    Super,
    Print,
    Channel,
    Broadcast,
    Stream,
    Stun,
    Relay,
    Signal,
    Connect,
    Block,
    Include
};

struct ASTNode {
    NodeKind kind;
    int line;
    int col;
    virtual ~ASTNode() = default;
    ASTNode(NodeKind k, int l = 0, int c = 0) : kind(k), line(l), col(c) {}
};

// The tag comes from the base-class name, so the struct's declaration line is
// the single place it is written and there is nothing to keep in sync.
template <NodeKind K>
struct ASTNodeK : ASTNode {
    static constexpr NodeKind kKind = K;
    ASTNodeK(int l = 0, int c = 0) : ASTNode(K, l, c) {}
};

// Downcast a node whose kind has already been established -- inside a switch
// arm, or after a nodeIf() succeeded. A plain static_cast is the whole point:
// it is what makes dispatch a jump table instead of a hierarchy walk.
//
// Built with -DBANTU_CHECK_NODEKIND it verifies the tag against RTTI and aborts
// naming both types. The full test suite is run that way as a phase gate, so a
// tag that ever disagreed with its type would have to get past every test in
// the tree before it could reach a release binary.
template <class T>
inline T* nodeAs(ASTNode* n) {
#ifdef BANTU_CHECK_NODEKIND
    if (n && !dynamic_cast<T*>(n)) {
        std::fprintf(stderr,
            "\n[BANTU] AST tag mismatch: node tagged %d is really %s, cast to %s\n",
            (int)n->kind, typeid(*n).name(), typeid(T).name());
        std::abort();
    }
#endif
    return static_cast<T*>(n);
}

// "Is this node a T?" -- the tag-comparing replacement for the scattered
// dynamic_cast<XNode*>(...) tests outside the dispatcher, several of which sit
// on hot paths of their own (resolveLValue, borrowLValue, evalCall, the
// assignment arms).
// The dispatcher's cast. The kind is passed as a template argument taken from
// the switch's own case label, and checked against the kind the target type
// declares -- so putting a node under the wrong case is a COMPILE error, not
// something a test has to notice. Together with the tag coming from the
// base-class name (ASTNodeK<K>), that leaves no way to express a mismatch.
template <NodeKind K, class T>
inline T* nodeExact(ASTNode* n) {
    static_assert(T::kKind == K,
        "evalNode: this case label does not match the node type it casts to");
    return nodeAs<T>(n);
}

template <class T>
inline T* nodeIf(ASTNode* n) {
    return (n && n->kind == T::kKind) ? nodeAs<T>(n) : nullptr;
}
template <class T>
inline T* nodeIf(const std::shared_ptr<ASTNode>& n) { return nodeIf<T>(n.get()); }

// ─── Literals ───
struct NumberNode : ASTNodeK<NodeKind::Number> {
    double value;
    NumberNode(double v, int l, int c) : ASTNodeK(l, c), value(v) {}
};

struct StringNode : ASTNodeK<NodeKind::String> {
    std::string value;
    StringNode(const std::string& v, int l, int c) : ASTNodeK(l, c), value(v) {}
};

struct BoolNode : ASTNodeK<NodeKind::Bool> {
    bool value;
    BoolNode(bool v, int l, int c) : ASTNodeK(l, c), value(v) {}
};

struct NullNode : ASTNodeK<NodeKind::Null> {
    NullNode(int l, int c) : ASTNodeK(l, c) {}
};

struct ListNode : ASTNodeK<NodeKind::List> {
    std::vector<std::shared_ptr<ASTNode>> elements;
    ListNode(std::vector<std::shared_ptr<ASTNode>> e, int l, int c) : ASTNodeK(l, c), elements(std::move(e)) {}
};

struct DictNode : ASTNodeK<NodeKind::Dict> {
    std::vector<std::pair<std::string, std::shared_ptr<ASTNode>>> pairs;
    DictNode(std::vector<std::pair<std::string, std::shared_ptr<ASTNode>>> p, int l, int c) : ASTNodeK(l, c), pairs(std::move(p)) {}
};

// ─── Variables ───
struct VariableNode : ASTNodeK<NodeKind::Variable> {
    std::string name;
    VariableNode(const std::string& n, int l, int c) : ASTNodeK(l, c), name(n) {}
};

struct VarDeclNode : ASTNodeK<NodeKind::VarDecl> {
    std::string typeAnnotation;
    std::string name;
    std::shared_ptr<ASTNode> init;
    VarDeclNode(const std::string& type, const std::string& n, std::shared_ptr<ASTNode> i, int l, int c)
        : ASTNodeK(l, c), typeAnnotation(type), name(n), init(std::move(i)) {}
};

struct AssignNode : ASTNodeK<NodeKind::Assign> {
    std::string name;
    std::shared_ptr<ASTNode> value;
    AssignNode(const std::string& n, std::shared_ptr<ASTNode> v, int l, int c)
        : ASTNodeK(l, c), name(n), value(std::move(v)) {}
    // Set by parseExpressionStatement for a bare `$x = …;`. Lets evalAssign
    // skip producing a value nobody reads -- which for the in-place string
    // append is the difference between O(n) and O(n^2), since the value it
    // would produce is a copy of the whole accumulated string.
    bool resultDiscarded = false;

    // Memo for evalAssign's in-place-append test: is this `$x = $x + …`, with
    // operands that cannot rebind $x? -1 not yet computed, 0 no, and otherwise
    // the number of pieces in the + chain. A pure function of the tree, which
    // never changes, so one spine walk at the first visit replaces one on every
    // visit -- and every `$i = $i + 1` in every loop in the language runs this.
    // Carrying the COUNT lets the single-piece case, which is nearly all of
    // them, run without allocating anything.
    signed char appendShape = -1;
};

// ─── Operations ───
struct BinaryOpNode : ASTNodeK<NodeKind::BinaryOp> {
    BantuTokenType op;
    std::shared_ptr<ASTNode> left, right;
    BinaryOpNode(BantuTokenType o, std::shared_ptr<ASTNode> l, std::shared_ptr<ASTNode> r, int ln, int c)
        : ASTNodeK(ln, c), op(o), left(std::move(l)), right(std::move(r)) {}
};

struct UnaryOpNode : ASTNodeK<NodeKind::UnaryOp> {
    BantuTokenType op;
    std::shared_ptr<ASTNode> operand;
    UnaryOpNode(BantuTokenType o, std::shared_ptr<ASTNode> e, int l, int c)
        : ASTNodeK(l, c), op(o), operand(std::move(e)) {}
};

// ─── Control Flow ───
struct IfNode : ASTNodeK<NodeKind::If> {
    std::shared_ptr<ASTNode> condition;
    std::vector<std::shared_ptr<ASTNode>> body;
    std::vector<std::shared_ptr<ASTNode>> elseBody;
    IfNode(std::shared_ptr<ASTNode> cond, std::vector<std::shared_ptr<ASTNode>> b,
           std::vector<std::shared_ptr<ASTNode>> eb, int l, int c)
        : ASTNodeK(l, c), condition(std::move(cond)), body(std::move(b)), elseBody(std::move(eb)) {}
};

struct WhileNode : ASTNodeK<NodeKind::While> {
    std::shared_ptr<ASTNode> condition;
    std::vector<std::shared_ptr<ASTNode>> body;
    WhileNode(std::shared_ptr<ASTNode> cond, std::vector<std::shared_ptr<ASTNode>> b, int l, int c)
        : ASTNodeK(l, c), condition(std::move(cond)), body(std::move(b)) {}
};

struct ForNode : ASTNodeK<NodeKind::For> {
    std::shared_ptr<ASTNode> init;
    std::shared_ptr<ASTNode> condition;
    std::shared_ptr<ASTNode> update;
    std::vector<std::shared_ptr<ASTNode>> body;
    ForNode(std::shared_ptr<ASTNode> i, std::shared_ptr<ASTNode> c, std::shared_ptr<ASTNode> u,
            std::vector<std::shared_ptr<ASTNode>> b, int l, int col)
        : ASTNodeK(l, col), init(std::move(i)), condition(std::move(c)), update(std::move(u)), body(std::move(b)) {}
};

struct EachNode : ASTNodeK<NodeKind::Each> {
    std::string varName;
    std::string valueVar;   // optional 2nd loop var for `for $k, $v in ...` ("" = single-var)
    std::shared_ptr<ASTNode> iterable;
    std::vector<std::shared_ptr<ASTNode>> body;
    EachNode(const std::string& v, std::shared_ptr<ASTNode> it, std::vector<std::shared_ptr<ASTNode>> b,
             int l, int c, const std::string& vv = "")
        : ASTNodeK(l, c), varName(v), valueVar(vv), iterable(std::move(it)), body(std::move(b)) {}
};

// ─── Loop control (break / continue) ───
struct BreakNode : ASTNodeK<NodeKind::Break> {
    BreakNode(int l, int c) : ASTNodeK(l, c) {}
};
struct ContinueNode : ASTNodeK<NodeKind::Continue> {
    ContinueNode(int l, int c) : ASTNodeK(l, c) {}
};

// ─── throw <expr>; ───
struct ThrowNode : ASTNodeK<NodeKind::Throw> {
    std::shared_ptr<ASTNode> value;
    ThrowNode(std::shared_ptr<ASTNode> v, int l, int c) : ASTNodeK(l, c), value(std::move(v)) {}
};

// ─── switch / case / default (no fallthrough) ───
struct SwitchCase {
    std::shared_ptr<ASTNode> value;                // case <value> { ... }
    std::vector<std::shared_ptr<ASTNode>> body;
};
struct SwitchNode : ASTNodeK<NodeKind::Switch> {
    std::shared_ptr<ASTNode> subject;
    std::vector<SwitchCase> cases;
    std::vector<std::shared_ptr<ASTNode>> defaultBody;
    bool hasDefault = false;
    SwitchNode(std::shared_ptr<ASTNode> s, std::vector<SwitchCase> cs,
               std::vector<std::shared_ptr<ASTNode>> db, bool hd, int l, int c)
        : ASTNodeK(l, c), subject(std::move(s)), cases(std::move(cs)),
          defaultBody(std::move(db)), hasDefault(hd) {}
};

// ─── Functions ───
struct FuncDeclNode : ASTNodeK<NodeKind::FuncDecl> {
    std::string name;
    std::vector<std::string> params;
    std::vector<std::shared_ptr<ASTNode>> body;
    FuncDeclNode(const std::string& n, std::vector<std::string> p, std::vector<std::shared_ptr<ASTNode>> b, int l, int c)
        : ASTNodeK(l, c), name(n), params(std::move(p)), body(std::move(b)) {}
};

struct ReturnNode : ASTNodeK<NodeKind::Return> {
    std::shared_ptr<ASTNode> value;
    ReturnNode(std::shared_ptr<ASTNode> v, int l, int c) : ASTNodeK(l, c), value(std::move(v)) {}
};

struct CallNode : ASTNodeK<NodeKind::Call> {
    std::shared_ptr<ASTNode> callee;
    std::vector<std::shared_ptr<ASTNode>> args;
    // Set by parseExpressionStatement when the call IS the whole statement, so
    // its result is thrown away. Only `push` consults it: returning the mutated
    // list means deep-copying every element, which turns an O(1) append into
    // O(n) and a loop of appends into O(n^2). Knowing the value is unused lets
    // push skip the copy while `$x = push($x, v)` still gets its list back.
    bool resultDiscarded = false;
    CallNode(std::shared_ptr<ASTNode> c, std::vector<std::shared_ptr<ASTNode>> a, int l, int col)
        : ASTNodeK(l, col), callee(std::move(c)), args(std::move(a)) {}
};

// ─── Property Access ───
struct DotAccessNode : ASTNodeK<NodeKind::DotAccess> {
    std::shared_ptr<ASTNode> object;
    std::string property;
    DotAccessNode(std::shared_ptr<ASTNode> o, const std::string& p, int l, int c)
        : ASTNodeK(l, c), object(std::move(o)), property(p) {}
};

struct IndexAccessNode : ASTNodeK<NodeKind::IndexAccess> {
    std::shared_ptr<ASTNode> object;
    std::shared_ptr<ASTNode> index;
    IndexAccessNode(std::shared_ptr<ASTNode> o, std::shared_ptr<ASTNode> i, int l, int c)
        : ASTNodeK(l, c), object(std::move(o)), index(std::move(i)) {}
};

struct IndexAssignNode : ASTNodeK<NodeKind::IndexAssign> {
    std::shared_ptr<ASTNode> object;
    std::shared_ptr<ASTNode> index;
    std::shared_ptr<ASTNode> value;
    IndexAssignNode(std::shared_ptr<ASTNode> o, std::shared_ptr<ASTNode> i, std::shared_ptr<ASTNode> v, int l, int c)
        : ASTNodeK(l, c), object(std::move(o)), index(std::move(i)), value(std::move(v)) {}

    // Memo for evalAssign's in-place-append test -- see AssignNode::appendShape.
    signed char appendShape = -1;
    bool resultDiscarded = false;
};

struct DictAssignNode : ASTNodeK<NodeKind::DictAssign> {
    std::shared_ptr<ASTNode> object;
    std::string key;
    std::shared_ptr<ASTNode> value;
    DictAssignNode(std::shared_ptr<ASTNode> o, const std::string& k, std::shared_ptr<ASTNode> v, int l, int c)
        : ASTNodeK(l, c), object(std::move(o)), key(k), value(std::move(v)) {}

    // Memo for evalAssign's in-place-append test -- see AssignNode::appendShape.
    signed char appendShape = -1;
    bool resultDiscarded = false;
};

// ─── Try-Catch ───
struct TryCatchNode : ASTNodeK<NodeKind::TryCatch> {
    std::vector<std::shared_ptr<ASTNode>> tryBody;
    std::string catchVar;
    std::vector<std::shared_ptr<ASTNode>> catchBody;
    TryCatchNode(std::vector<std::shared_ptr<ASTNode>> tb, const std::string& cv,
                 std::vector<std::shared_ptr<ASTNode>> cb, int l, int c)
        : ASTNodeK(l, c), tryBody(std::move(tb)), catchVar(cv), catchBody(std::move(cb)) {}
};

// ─── Class Declaration ───
struct ClassDeclNode : ASTNodeK<NodeKind::ClassDecl> {
    std::string name;
    std::string parentClass;  // extends (single inheritance)
    std::vector<std::string> implementsClasses;  // implements (multiple inheritance)
    std::vector<std::shared_ptr<ASTNode>> body;  // method definitions
    ClassDeclNode(const std::string& n, const std::string& parent,
                  std::vector<std::string> impl,
                  std::vector<std::shared_ptr<ASTNode>> b, int l, int c)
        : ASTNodeK(l, c), name(n), parentClass(parent),
          implementsClasses(std::move(impl)), body(std::move(b)) {}
};

// ─── Super Call ───
struct SuperNode : ASTNodeK<NodeKind::Super> {
    std::vector<std::shared_ptr<ASTNode>> args;
    SuperNode(std::vector<std::shared_ptr<ASTNode>> a, int l, int c)
        : ASTNodeK(l, c), args(std::move(a)) {}
};

// ─── Print ───
struct PrintNode : ASTNodeK<NodeKind::Print> {
    std::shared_ptr<ASTNode> value;
    PrintNode(std::shared_ptr<ASTNode> v, int l, int c) : ASTNodeK(l, c), value(std::move(v)) {}
};

// ─── sua Framework Nodes ───

// sua.channel("name", callback)
struct ChannelNode : ASTNodeK<NodeKind::Channel> {
    std::string channelName;
    std::shared_ptr<ASTNode> callback;
    ChannelNode(const std::string& name, std::shared_ptr<ASTNode> cb, int l, int c)
        : ASTNodeK(l, c), channelName(name), callback(std::move(cb)) {}
};

// sua.broadcast("channel", message)
struct BroadcastNode : ASTNodeK<NodeKind::Broadcast> {
    std::string channelName;
    std::shared_ptr<ASTNode> message;
    BroadcastNode(const std::string& name, std::shared_ptr<ASTNode> msg, int l, int c)
        : ASTNodeK(l, c), channelName(name), message(std::move(msg)) {}
};

// sua.stream("channel", "video", callback)
struct StreamNode : ASTNodeK<NodeKind::Stream> {
    std::string channelName;
    std::string streamType;
    std::shared_ptr<ASTNode> callback;
    StreamNode(const std::string& ch, const std::string& type, std::shared_ptr<ASTNode> cb, int l, int c)
        : ASTNodeK(l, c), channelName(ch), streamType(type), callback(std::move(cb)) {}
};

// sua.stun() -> dict
struct StunNode : ASTNodeK<NodeKind::Stun> {
    StunNode(int l, int c) : ASTNodeK(l, c) {}
};

// sua.relay("peerId") -> dict
struct RelayNode : ASTNodeK<NodeKind::Relay> {
    std::shared_ptr<ASTNode> peerId;
    RelayNode(std::shared_ptr<ASTNode> pid, int l, int c)
        : ASTNodeK(l, c), peerId(std::move(pid)) {}
};

// sua.signal(peerId, callback)
struct SignalNode : ASTNodeK<NodeKind::Signal> {
    std::shared_ptr<ASTNode> targetPeer;
    std::shared_ptr<ASTNode> callback;
    SignalNode(std::shared_ptr<ASTNode> target, std::shared_ptr<ASTNode> cb, int l, int c)
        : ASTNodeK(l, c), targetPeer(std::move(target)), callback(std::move(cb)) {}
};

// sua.connect("peerId")
struct ConnectNode : ASTNodeK<NodeKind::Connect> {
    std::shared_ptr<ASTNode> peerId;
    ConnectNode(std::shared_ptr<ASTNode> pid, int l, int c)
        : ASTNodeK(l, c), peerId(std::move(pid)) {}
};

// ─── Block ───
struct BlockNode : ASTNodeK<NodeKind::Block> {
    std::vector<std::shared_ptr<ASTNode>> statements;
    BlockNode(std::vector<std::shared_ptr<ASTNode>> s, int l, int c) : ASTNodeK(l, c), statements(std::move(s)) {}
};

// ─── Include (v1.2.1 module system) ───
//   include "./routes.b";
//   include "./controller.b" as ctrl;
struct IncludeNode : ASTNodeK<NodeKind::Include> {
    std::string path;          // resolved relative path
    std::string alias;         // empty = direct (symbols into current scope); non-empty = namespaced
    IncludeNode(std::string p, std::string a, int l, int c)
        : ASTNodeK(l, c), path(std::move(p)), alias(std::move(a)) {}
};
