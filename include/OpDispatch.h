// OpDispatch.h
#pragma once
#include "ADGraph.h"
#include "OpTraits.h"

struct ForwardFunctor {
    ADGraph& g; ADNode& n;
    template<Operator Op> inline void operator()() { OpTraits<Op>::forward(n,g); }
};
struct ForwardDotFunctor {
    ADGraph& g; ADNode& n;
    template<Operator Op> inline void operator()() { OpTraits<Op>::forward_dot(n,g); }
};
struct BackwardFunctor {
    ADGraph& g; ADNode& n;
    template<Operator Op> inline void operator()() { OpTraits<Op>::backward(n,g); }
};
struct HVPBackwardFunctor {
    ADGraph& g; ADNode& n;
    template<Operator Op> inline void operator()() { OpTraits<Op>::hvp_backward(n,g); }
};

template<typename Fn>
inline void _dispatch_op(ADNode& n, Fn&& fn){
    switch(n.type){
        case Operator::Var:      fn.template operator()<Operator::Var>(); break;
        case Operator::cte:      fn.template operator()<Operator::cte>(); break;
        case Operator::Sin:      fn.template operator()<Operator::Sin>(); break;
        case Operator::Cos:      fn.template operator()<Operator::Cos>(); break;
        case Operator::Tan:      fn.template operator()<Operator::Tan>(); break;
        case Operator::Exp:      fn.template operator()<Operator::Exp>(); break;
        case Operator::Log:      fn.template operator()<Operator::Log>(); break;
        case Operator::Add:      fn.template operator()<Operator::Add>(); break;
        case Operator::Subtract: fn.template operator()<Operator::Subtract>(); break;
        case Operator::Multiply: fn.template operator()<Operator::Multiply>(); break;
        case Operator::Divide:   fn.template operator()<Operator::Divide>(); break;
        case Operator::Max:      fn.template operator()<Operator::Max>(); break;
        default: /* Unknown/NA */ break;
    }
}
