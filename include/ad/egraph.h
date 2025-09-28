// egraph.hpp
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <unordered_map>
#include <utility>
#include <vector>

// You already have Operator in your codebase.
extern "C++" {
enum class Operator; // forward, defined in your AD headers
}

// -------- Budget / Config --------
struct EGraphBudget {
    int max_iterations = 5;
    size_t max_nodes = 100000;    // global arena hard limit
    int max_distribute_terms = 4; // cap fanout for x*(y1+...+yk)
};

// -------- E-graph atoms --------
struct EClassId {
    int id = -1;
    bool operator==(const EClassId &o) const { return id == o.id; }
};
struct ENode {
    Operator op{};
    std::vector<EClassId> kids;
    bool is_const = false;
    double cval = 0.0;
    bool is_symbol =
        false; // terminals bound to an external symbol (e.g., ADNode*)
    const void *sym_ptr = nullptr; // pointer identity for terminals (ADNode*)
};
struct EClass {
    std::vector<int> nodes; // indices into arena
    // simple analysis facts
    bool has_const = false;
    double cval = 0.0;
    bool nonzero = false;
    bool positive = false;
};

struct EGraph {
    std::vector<ENode> arena;
    std::vector<EClass> classes;

    // union-find (per e-node; parent over e-class ids)
    std::vector<int> parent;              // index by e-class id
    std::unordered_map<size_t, int> memo; // hash(ENode)->eclass id
    // --- before: existing non-const find ---
    int find(int x) {
        if (parent[x] == x)
            return x;
        parent[x] = find(parent[x]); // path compression
        return parent[x];
    }

    // --- add: const overload (no compression) ---
    int find(int x) const {
        // Walk to representative without modifying parent[]
        while (parent[x] != x)
            x = parent[x];
        return x;
    }

    // --- before: canonicalize_kids non-const ---
    void canonicalize_kids(ENode &n) {
        for (auto &k : n.kids)
            k.id = find(k.id);
    }

    // --- add: const overload for const contexts (e.g., hashing, costing) ---
    void canonicalize_kids(ENode &n) const {
        for (auto &k : n.kids)
            k.id = find(k.id);
    }

    void unite(int a, int b) {
        a = find(a);
        b = find(b);
        if (a == b)
            return;
        parent[b] = a;
        // merge analysis facts conservatively
        classes[a].has_const = classes[a].has_const || classes[b].has_const;
        if (classes[b].has_const)
            classes[a].cval = classes[b].cval;
        classes[a].nonzero |= classes[b].nonzero;
        classes[a].positive |= classes[b].positive;
        // append nodes
        for (int n : classes[b].nodes)
            classes[a].nodes.push_back(n);
        classes[b].nodes.clear();
    }



    // structural hash; for Add/Mul we assume kids are already flattened+sorted
    // (bridge)
    static size_t hash_node(const ENode &n) {
        auto mix = [](size_t a, size_t b) {
            return a ^ (b + 0x9e3779b97f4a7c15ULL + (a << 6) + (a >> 2));
        };
        size_t h = std::hash<int>{}(int(n.op));
        h = mix(h, std::hash<bool>{}(n.is_const));
        if (n.is_const) {
            uint64_t bits;
            std::memcpy(&bits, &n.cval, sizeof(bits));
            h = mix(h, std::hash<uint64_t>{}(bits));
        }
        h = mix(h, std::hash<bool>{}(n.is_symbol));
        if (n.is_symbol)
            h = mix(h, std::hash<const void *>{}(n.sym_ptr));
        for (auto k : n.kids)
            h = mix(h, std::hash<int>{}(k.id));
        return h;
    }

    int new_eclass() {
        int id = (int)classes.size();
        classes.push_back(EClass{});
        parent.push_back(id);
        return id;
    }

    // “add” inserts an enode into the egraph, merging with congruent nodes
    int add(const ENode &raw) {
        ENode n = raw;
        canonicalize_kids(n);
        size_t h = hash_node(n);
        auto it = memo.find(h);
        if (it != memo.end()) {
            // congruent enode already exists; attach in that eclass
            int ec = find(it->second);
            int idx = (int)arena.size();
            arena.push_back(n);
            classes[ec].nodes.push_back(idx);
            // update easy facts
            if (n.is_const) {
                classes[ec].has_const = true;
                classes[ec].cval = n.cval;
            }
            return ec;
        }
        // create new eclass
        int ec = new_eclass();
        int idx = (int)arena.size();
        arena.push_back(n);
        classes[ec].nodes.push_back(idx);
        memo[h] = ec;
        if (n.is_const) {
            classes[ec].has_const = true;
            classes[ec].cval = n.cval;
        }
        return ec;
    }

    // merge two existing eclasses
    void merge(int a, int b) { unite(a, b); }

    // congruence closure rebuild: re-hash with canonical reps, merging if
    // needed
    void rebuild() {
        std::unordered_map<size_t, int> fresh;
        for (int ec = 0; ec < (int)classes.size(); ++ec) {
            if (classes[ec].nodes.empty())
                continue;
            int rep = find(ec);
            if (rep != ec)
                continue;
            for (int idx : classes[ec].nodes) {
                ENode n = arena[idx];
                canonicalize_kids(n);
                size_t h = hash_node(n);
                auto it = fresh.find(h);
                if (it == fresh.end()) {
                    fresh[h] = rep;
                } else {
                    unite(rep, it->second);
                }
            }
        }
        memo.swap(fresh);
    }
};

// ---------- Tiny cost model ----------
struct ECostModel {
    int op_cost(Operator op) const;
    // returns (best_cost, best_node_index_in_class)
    std::pair<int, int> best_of(int eclass, const EGraph &G,
                                std::unordered_map<int, int> &memo) const;
};
