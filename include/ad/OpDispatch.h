// OpDispatch.h - Updated with Lane Support
#pragma once
#include "ADGraph.h"
#include "OpTraits.h"

// Existing functors
struct ForwardFunctor {
    ADGraph& g; ADNode& n;
    template<Operator Op> inline void operator()() { OpTraits<Op>::forward(n, g); }
};

struct ForwardDotFunctor {
    ADGraph& g; ADNode& n;
    template<Operator Op> inline void operator()() { OpTraits<Op>::forward_dot(n, g); }
};

struct BackwardFunctor {
    ADGraph& g; ADNode& n;
    template<Operator Op> inline void operator()() { OpTraits<Op>::backward(n, g); }
};

struct HVPBackwardFunctor {
    ADGraph& g; ADNode& n;
    template<Operator Op> inline void operator()() { OpTraits<Op>::hvp_backward(n, g); }
};

// NEW: Lane-aware functors
struct ForwardDotLanesFunctor {
    ADGraph& g; 
    ADNode& n;
    size_t L;      // lane count
    size_t ybase;  // output base index
    
    ForwardDotLanesFunctor(ADGraph& graph, ADNode& node, size_t lanes, size_t base) 
        : g(graph), n(node), L(lanes), ybase(base) {}
    
    template<Operator Op> 
    inline void operator()() { 
        OpTraits<Op>::forward_dot_lanes(n, g, L, ybase); 
    }
};

struct BackwardLanesFunctor {
    ADGraph& g; 
    ADNode& n;
    size_t L;      // lane count
    size_t ybase;  // output base index
    
    BackwardLanesFunctor(ADGraph& graph, ADNode& node, size_t lanes, size_t base) 
        : g(graph), n(node), L(lanes), ybase(base) {}
    
    template<Operator Op> 
    inline void operator()() { 
        OpTraits<Op>::backward_lanes(n, g, L, ybase); 
    }
};

struct FusedForwardFunctor {
    ADGraph& g; 
    ADNode& n;
    size_t L;      // lane count
    size_t ybase;  // output base index
    
    FusedForwardFunctor(ADGraph& graph, ADNode& node, size_t lanes, size_t base) 
        : g(graph), n(node), L(lanes), ybase(base) {}
    
    template<Operator Op> 
    inline void operator()() { 
        OpTraits<Op>::fused_forward(n, g, L, ybase); 
    }
};

// Updated dispatch function (unchanged)
template<typename Fn>
inline void dispatch_op(ADNode& n, Fn&& fn){
    switch(n.type){
        case Operator::Var: fn.template operator()<Operator::Var>(); break;
        case Operator::cte: fn.template operator()<Operator::cte>(); break;
        case Operator::Sin: fn.template operator()<Operator::Sin>(); break;
        case Operator::Cos: fn.template operator()<Operator::Cos>(); break;
        case Operator::Tan: fn.template operator()<Operator::Tan>(); break;
        case Operator::Exp: fn.template operator()<Operator::Exp>(); break;
        case Operator::Log: fn.template operator()<Operator::Log>(); break;
        case Operator::Add: fn.template operator()<Operator::Add>(); break;
        case Operator::Subtract: fn.template operator()<Operator::Subtract>(); break;
        case Operator::Multiply: fn.template operator()<Operator::Multiply>(); break;
        case Operator::Divide: fn.template operator()<Operator::Divide>(); break;
        case Operator::Max: fn.template operator()<Operator::Max>(); break;
        case Operator::Tanh: fn.template operator()<Operator::Tanh>(); break;
        case Operator::Silu: fn.template operator()<Operator::Silu>(); break;
        case Operator::Gelu: fn.template operator()<Operator::Gelu>(); break;
        case Operator::Relu: fn.template operator()<Operator::Relu>(); break;
        case Operator::Softmax: fn.template operator()<Operator::Softmax>(); break;
        default: /* Unknown/NA */ break;
    }
}