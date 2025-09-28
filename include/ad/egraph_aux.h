
// egraph_rules.hpp
#pragma once
#include "Definitions.h"
#include "egraph.h"
#include "ADGraph.h"
#include <queue>

// Forward decls from your codebase
struct ADNode;
using ADNodePtr = std::shared_ptr<ADNode>;
struct ADGraph;

// Small factory hooks you add to ADGraph (shown below patch)
ADNodePtr AD_makeNode(ADGraph& G, Operator op, const std::vector<ADNodePtr>& kids);
ADNodePtr AD_makeConst(ADGraph& G, double v);

// Flatten + sort for AC ops
static void flatten_AC_(Operator op, const ADNodePtr& n, std::vector<ADNodePtr>& out);

// Build EGraph from an AD subtree; returns e-class id
static int to_egraph_rec(const ADNodePtr& n, EGraph& EG) {
    // Constants
    if (n->type == Operator::cte) {
        ENode c; c.op=Operator::cte; c.is_const=true; c.cval=n->value;
        return EG.add(c);
    }

    // Terminals (variables / symbols) — use pointer identity
    if (n->inputs.empty() && n->type != Operator::cte) {
        ENode t; t.op=n->type; t.is_symbol=true; t.sym_ptr=(const void*)n.get();
        return EG.add(t);
    }

    // General case
    ENode e; e.op = n->type;
    std::vector<ADNodePtr> kids;
    if (n->type==Operator::Add || n->type==Operator::Multiply) {
        flatten_AC_(n->type, n, kids); // n-ary flatten
        // translate each child
        for (auto& k : kids) e.kids.push_back({ to_egraph_rec(k, EG) });
        // sort for AC canonical form
        std::sort(e.kids.begin(), e.kids.end(), [](auto a, auto b){return a.id<b.id;});
        return EG.add(e);
    } else {
        for (auto& k : n->inputs) e.kids.push_back({ to_egraph_rec(k, EG) });
        return EG.add(e);
    }
}

inline void flatten_AC_(Operator op, const ADNodePtr& n, std::vector<ADNodePtr>& out) {
    if (n->type==op) {
        for (auto& c : n->inputs) flatten_AC_(op, c, out);
    } else {
        out.push_back(n);
    }
}

// Extraction: pick cheapest enode in each class, then rebuild AD nodes
static ADNodePtr extract_rec(int eclass, const EGraph& EG, ECostModel& cm,
                             std::unordered_map<int,ADNodePtr>& memo, ADGraph& G) {
    int rep = EG.find(eclass);
    auto it = memo.find(rep);
    if (it!=memo.end()) return it->second;

    std::unordered_map<int,int> costmemo;
    auto best = cm.best_of(rep, EG, costmemo); // (cost, idx)
    const ENode& n = EG.arena[best.second];

    if (n.is_const) {
        auto r = AD_makeConst(G, n.cval);
        memo[rep] = r; return r;
    }
    if (n.is_symbol) {
        // Refer back to original AD node by pointer; safe to wrap without copy.
        ADNode* orig = (ADNode*)n.sym_ptr;
        // Find or recreate a shared_ptr wrapper (we assume your graph retains ownership)
        // Fast path: your ADGraph keeps stable ADNodePtr; we reuse via id map if needed.
        // For now, we create a lightweight aliasing shared_ptr (no deleter).
        ADNodePtr sptr(orig, [](ADNode*){}); // aliasing, no ownership transfer
        memo[rep] = sptr; return sptr;
    }

    std::vector<ADNodePtr> kids;
    kids.reserve(n.kids.size());
    for (auto k : n.kids) kids.push_back(extract_rec(k.id, EG, cm, memo, G));
    ADNodePtr built = AD_makeNode(G, n.op, kids);
    memo[rep] = built;
    return built;
}

struct EGraphBridge {
    static int to_egraph(const ADNodePtr& root, EGraph& EG) {
        return to_egraph_rec(root, EG);
    }
    static ADNodePtr extract_expr(int root_ec, const EGraph& EG, ADGraph& G) {
        ECostModel cm;
        std::unordered_map<int,ADNodePtr> memo;
        return extract_rec(root_ec, EG, cm, memo, G);
    }
};


// Helper: quick float quantization for stable zero/one checks
static inline double qfp_(double x, double s=1e12){ return std::nearbyint(x*s)/s; }
static inline bool is_zero_(double x){ return qfp_(x)==0.0; }
static inline bool is_one_(double x){ return qfp_(x)==1.0; }

// NOTE: Real AC is hard; we fake it by flattening/sorting in the bridge.
// Rules here assume Add/Multiply nodes are already flattened (n-ary).

// ---- Rule applications mutate egraph by *adding* canonical nodes and merging ----

static bool rule_add_zero_(EGraph& G, int ec) {
    // If Add contains zeros, drop them; if only one child remains, merge to it.
    bool changed=false;
    // Scan each representative node of this class
    int rep = G.find(ec);
    for (int idx : G.classes[rep].nodes) {
        const ENode& n = G.arena[idx];
        if (n.op != Operator::Add || n.kids.size()<2) continue;
        std::vector<EClassId> keep;
        for (auto k : n.kids) {
            int kc = G.find(k.id);
            if (G.classes[kc].has_const && is_zero_(G.classes[kc].cval)) {
                changed = true; continue;
            }
            keep.push_back({kc});
        }
        if (!changed) continue;
        if (keep.empty()) {
            // sum of zeros -> 0
            ENode c; c.op=Operator::cte; c.is_const=true; c.cval=0.0;
            int z = G.add(c);
            G.merge(rep, z);
        } else if (keep.size()==1) {
            G.merge(rep, keep[0].id);
        } else {
            ENode m; m.op=Operator::Add; m.kids = keep;
            std::sort(m.kids.begin(), m.kids.end(), [](auto a, auto b){return a.id<b.id;});
            int e2 = G.add(m);
            G.merge(rep, e2);
        }
        break; // one application per rebuild is enough
    }
    return changed;
}

static bool rule_mul_one_zero_(EGraph& G, int ec) {
    bool changed=false;
    int rep = G.find(ec);
    for (int idx : G.classes[rep].nodes) {
        const ENode& n = G.arena[idx];
        if (n.op != Operator::Multiply || n.kids.empty()) continue;

        bool has_zero=false;
        std::vector<EClassId> keep;
        for (auto k : n.kids) {
            int kc = G.find(k.id);
            if (G.classes[kc].has_const) {
                if (is_zero_(G.classes[kc].cval)) { has_zero=true; break; }
                if (is_one_(G.classes[kc].cval)) { changed=true; continue; } // drop 1
            }
            keep.push_back({kc});
        }
        if (has_zero) {
            ENode c; c.op=Operator::cte; c.is_const=true; c.cval=0.0;
            G.merge(rep, G.add(c));
            return true;
        }
        if (!changed) continue;
        if (keep.empty()) {
            ENode c; c.op=Operator::cte; c.is_const=true; c.cval=1.0;
            G.merge(rep, G.add(c));
        } else if (keep.size()==1) {
            G.merge(rep, keep[0].id);
        } else {
            ENode m; m.op=Operator::Multiply; m.kids=keep;
            std::sort(m.kids.begin(), m.kids.end(), [](auto a, auto b){return a.id<b.id;});
            G.merge(rep, G.add(m));
        }
        return true;
    }
    return false;
}

static bool rule_const_fold_add_mul_(EGraph& G, int ec) {
    bool changed=false;
    int rep = G.find(ec);
    for (int idx : G.classes[rep].nodes) {
        const ENode& n = G.arena[idx];
        if (n.op!=Operator::Add && n.op!=Operator::Multiply) continue;
        double acc = (n.op==Operator::Add)? 0.0 : 1.0;
        bool all_const = !n.kids.empty();
        for (auto k : n.kids) {
            int kc = G.find(k.id);
            if (!(G.classes[kc].has_const)) { all_const=false; break; }
            double v = G.classes[kc].cval;
            acc = (n.op==Operator::Add)? acc + v : acc * v;
        }
        if (all_const) {
            ENode c; c.op=Operator::cte; c.is_const=true; c.cval=acc;
            G.merge(rep, G.add(c));
            changed=true;
            break;
        }
    }
    return changed;
}

static bool rule_distribute_left_(EGraph& G, int ec, const EGraphBudget& budget) {
    // x * (y1 + ... + yk) -> x*y1 + ... + x*yk, k <= max_distribute_terms
    int rep = G.find(ec);
    for (int idx : G.classes[rep].nodes) {
        const ENode& n = G.arena[idx];
        if (n.op != Operator::Multiply || n.kids.size()!=2) continue;
        int L = G.find(n.kids[0].id);
        int R = G.find(n.kids[1].id);
        // Check if right is Add
        bool right_is_add = false; std::vector<EClassId> summands;
        for (int ridx : G.classes[R].nodes) {
            const ENode& rn = G.arena[ridx];
            if (rn.op==Operator::Add && rn.kids.size()>=2) {
                right_is_add = true;
                for (auto s : rn.kids) summands.push_back({G.find(s.id)});
                break;
            }
        }
        if (!right_is_add || (int)summands.size()>budget.max_distribute_terms) continue;
        // Build x*yi
        std::vector<EClassId> terms; terms.reserve(summands.size());
        for (auto s : summands) {
            ENode mul; mul.op=Operator::Multiply; mul.kids = { {L}, {s.id} };
            if (mul.kids[0].id > mul.kids[1].id) std::swap(mul.kids[0], mul.kids[1]);
            int m = G.add(mul);
            terms.push_back({m});
        }
        // Sum them
        ENode add; add.op=Operator::Add; add.kids = terms;
        std::sort(add.kids.begin(), add.kids.end(), [](auto a, auto b){return a.id<b.id;});
        int sumc = G.add(add);
        G.merge(rep, sumc);
        return true;
    }
    return false;
}

static bool rule_factor_common_(EGraph& G, int ec) {
    // x*y1 + x*y2 + ... -> x*(y1 + y2 + ...)
    int rep = G.find(ec);
    for (int idx : G.classes[rep].nodes) {
        const ENode& n = G.arena[idx];
        if (n.op != Operator::Add || n.kids.size()<2) continue;

        // Collect a candidate common factor from first summand if it’s a Multiply
        int first = G.find(n.kids[0].id);
        int cand = -1;
        for (int nidx : G.classes[first].nodes) {
            const ENode& mn = G.arena[nidx];
            if (mn.op==Operator::Multiply && mn.kids.size()>=2) {
                cand = G.find(mn.kids[0].id); // pick first factor as candidate
                break;
            }
        }
        if (cand<0) continue;

        // Verify all summands contain cand as a direct factor
        std::vector<EClassId> rest_terms;
        bool ok = true;
        for (auto s : n.kids) {
            int sc = G.find(s.id);
            bool found=false;
            for (int sidx : G.classes[sc].nodes) {
                const ENode& sn = G.arena[sidx];
                if (sn.op==Operator::Multiply) {
                    // split: cand * other or other * cand
                    std::vector<EClassId> others;
                    for (auto f : sn.kids) {
                        if (G.find(f.id)==cand) { found=true; }
                        else others.push_back({G.find(f.id)});
                    }
                    if (found) {
                        if (others.empty()) {
                            // term is exactly cand → contributes 1
                            ENode c; c.op=Operator::cte; c.is_const=true; c.cval=1.0;
                            rest_terms.push_back({G.add(c)});
                        } else if (others.size()==1) {
                            rest_terms.push_back({others[0].id});
                        } else {
                            ENode mul; mul.op=Operator::Multiply; mul.kids = others;
                            std::sort(mul.kids.begin(), mul.kids.end(), [](auto a, auto b){return a.id<b.id;});
                            rest_terms.push_back({G.add(mul)});
                        }
                        break;
                    }
                }
            }
            if (!found) { ok=false; break; }
        }
        if (!ok) continue;

        // Build (sum of rest) then multiply by cand
        ENode sum; sum.op=Operator::Add; sum.kids = rest_terms;
        std::sort(sum.kids.begin(), sum.kids.end(), [](auto a, auto b){return a.id<b.id;});
        int sumc = G.add(sum);

        ENode mul; mul.op=Operator::Multiply; mul.kids = { {cand}, {sumc} };
        if (mul.kids[0].id > mul.kids[1].id) std::swap(mul.kids[0], mul.kids[1]);
        int fac = G.add(mul);

        G.merge(rep, fac);
        return true;
    }
    return false;
}

// One saturation round over all classes
inline bool apply_rules_once(EGraph& G, const EGraphBudget& budget) {
    bool changed=false;
    for (int ec=0; ec<(int)G.classes.size(); ++ec) {
        if (G.classes[ec].nodes.empty()) continue;
        changed |= rule_add_zero_(G, ec);
        changed |= rule_mul_one_zero_(G, ec);
        changed |= rule_const_fold_add_mul_(G, ec);
        if (changed) continue; // rebuild soon
        changed |= rule_distribute_left_(G, ec, budget);
        if (changed) continue;
        changed |= rule_factor_common_(G, ec);
        if (changed) continue;
    }
    return changed;
}

// Driver: bounded saturation
inline void saturate(EGraph& G, const EGraphBudget& budget) {
    int it=0;
    for (; it<budget.max_iterations; ++it) {
        bool any = apply_rules_once(G, budget);
        G.rebuild();
        if (!any) break;
        if (G.arena.size() > budget.max_nodes) break;
    }
}
