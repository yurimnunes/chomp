from typing import Dict, List, Tuple

import numpy as np
from scipy.sparse import csr_matrix, isspmatrix_csr, triu


def inverse_permutation(p: np.ndarray) -> np.ndarray:
    inv = np.empty_like(p)
    inv[p] = np.arange(p.size, dtype=p.dtype)
    return inv


class AMDReorderingArray:
    """
    Array-based AMD with quotient-graph maintenance, featuring:
      - Undirected adjacency (both directions stored)
      - AMD-style approximate external degree (rolling w-flag)
      - Hashed element equality/subset absorption (SuiteSparse-style)
      - Variable supervariable coalescence (identical element sets)
      - Dynamic iw growth + periodic compaction
      - Dense postponement (heuristic cutoff)
      - Tie-breaking pivot selection (prefer larger nv, then smaller id)
      - Reverse(elimination order) as final permutation

    Variables: ids 0..n-1 (elen[i] == -1, len[i] used)
    Elements : ids >= n   (elen[e] >= 0, elen[e] used, len[e] unused)
    """

    def __init__(self, aggressive_absorption: bool = True, dense_cutoff: int | None = None):
        self.aggressive_absorption = aggressive_absorption

        # Problem size
        self.n = 0
        self.nsym = 0
        self.nzmax = 0

        # Core arrays
        self.pe = None
        self.len = None
        self.elen = None
        self.iw = None
        self.nv = None
        self.degree = None
        self.w = None
        self.last = None

        # Buckets
        self.head = None
        self.next = None
        self.prev = None
        self.where = None
        self.mindeg = 0

        # Activity flags
        self.var_active = None
        self.elem_active = None

        # Elements allocation
        self.nelem_top = 0

        # Ordering / outputs
        self.order: List[int] = []       # elimination order of ORIGINAL variables (expanded groups)
        self.perm = None
        self.dense_queue: List[int] = []

        # Tail bump pointer for iw
        self._tail_used = 0

        # Rolling mark flag
        self._wflg = 1

        # Supervariable tracking (lists of original vars per representative)
        self.sv_members: List[List[int]] = []

        # For coverage check
        self._in_order = None
        self.dense_cutoff = dense_cutoff  # None = heuristic; 0 = off; >0 = explicit threshold

    # ---------------- Public API ----------------

    def amd_order(self, A: csr_matrix, symmetrize: bool = True) -> np.ndarray:
        if not isspmatrix_csr(A):
            A = csr_matrix(A)

        # Structural symmetrization (or trust upper) and drop diagonal, keep CSR
        if symmetrize:
            A_up = triu(A + A.T, k=1, format="csr")
        else:
            A_up = triu(A, k=1, format="csr")
        A_up.eliminate_zeros()

        self.n = A_up.shape[0]
        if self.n == 0:
            return np.array([], dtype=np.int32)

        self._initialize_from_upper(A_up)
        self._initialize_buckets()
        self._eliminate_all()
        return self.perm.copy()

    def compute_fill_reducing_permutation(self, A: csr_matrix, symmetrize: bool = True) -> Tuple[np.ndarray, Dict]:
        orig_nnz = A.nnz
        orig_bw = self._bandwidth(A)
        p = self.amd_order(A, symmetrize=symmetrize)
        Ap = A[p, :][:, p]
        new_bw = self._bandwidth(Ap)
        stats = {
            "original_nnz": int(orig_nnz),
            "original_bandwidth": int(orig_bw),
            "reordered_bandwidth": int(new_bw),
            "bandwidth_reduction": 0.0 if orig_bw == 0 else float(orig_bw - new_bw) / float(orig_bw),
            "matrix_size": int(A.shape[0]),
            "inverse_permutation": inverse_permutation(p),
        }
        return p, stats

    # ---------------- Initialization ----------------

    def _initialize_from_upper(self, A_up: csr_matrix):
        """
        Build an undirected adjacency from a strictly-upper-triangular CSR matrix A_up.
        For every edge (i,j) with i<j, we insert j into i's list and i into j's list.
        """
        n = self.n
        indptr, indices = A_up.indptr, A_up.indices
        m = A_up.nnz

        # Need storage for both directions
        est_undirected_nnz = 2 * m

        # Start with a comfortable capacity; both iw and nsym can grow
        self.nzmax = max(est_undirected_nnz + n, int(1.5 * est_undirected_nnz) + 4 * n, 1)
        self.nsym = n + max(8, n)

        self.pe = np.zeros(self.nsym, dtype=np.int32)
        self.len = np.zeros(self.nsym, dtype=np.int32)
        self.elen = np.full(self.nsym, -2, dtype=np.int32)   # -2 = dead
        self.iw = np.zeros(self.nzmax, dtype=np.int32)
        self.nv = np.ones(self.nsym, dtype=np.int32)
        self.degree = np.zeros(self.nsym, dtype=np.int32)
        self.w = np.zeros(self.nsym, dtype=np.int32)
        self.last = np.full(self.nsym, -1, dtype=np.int32)

        self.var_active = np.zeros(self.nsym, dtype=np.int8)
        self.elem_active = np.zeros(self.nsym, dtype=np.int8)

        # ---- Pass 1: degrees (both directions) ----
        deg = np.zeros(n, dtype=np.int32)
        for i in range(n):
            s, e = indptr[i], indptr[i + 1]
            for p in range(s, e):
                j = indices[p]  # j > i guaranteed
                deg[i] += 1
                deg[j] += 1

        # ---- Compute pe/len from degrees ----
        pos = 0
        for i in range(n):
            self.pe[i] = pos
            self.len[i] = deg[i]
            self.degree[i] = deg[i]
            self.elen[i] = -1        # variable
            self.var_active[i] = 1
            self.elem_active[i] = 0
            pos += deg[i]

        # Ensure capacity
        if pos > self.nzmax:
            self._grow_iw(pos)

        # ---- Pass 2: fill adjacency (both directions) ----
        write_ptr = self.pe[:n].copy()
        for i in range(n):
            s, e = indptr[i], indptr[i + 1]
            for p in range(s, e):
                j = indices[p]  # i < j
                wi = write_ptr[i]
                wj = write_ptr[j]
                if wi >= self.nzmax or wj >= self.nzmax:
                    self._grow_iw(max(wi, wj) + 1)
                self.iw[wi] = j
                self.iw[wj] = i
                write_ptr[i] += 1
                write_ptr[j] += 1

        # Tail used is exactly the undirected nnz
        self._tail_used = pos

        # Element slots initialize to dead
        for e in range(n, self.nsym):
            self.pe[e] = -1
            self.len[e] = 0
            self.elen[e] = -2
            self.nv[e] = 0
            self.elem_active[e] = 0
            self.var_active[e] = 0

        self.nelem_top = n
        self.perm = np.full(n, -1, dtype=np.int32)
        self.order.clear()
        self.dense_queue.clear()
        self._wflg = 1

        # supervariable groups: each var starts as its own group
        self.sv_members = [[] for _ in range(n)]
        for i in range(n):
            self.sv_members[i] = [i]

        self._in_order = np.zeros(n, dtype=np.int8)

    def _initialize_buckets(self):
        n = self.n
        self.head = np.full(n + 1, -1, dtype=np.int32)
        self.next = np.full(self.nsym, -1, dtype=np.int32)
        self.prev = np.full(self.nsym, -1, dtype=np.int32)
        self.where = np.full(self.nsym, -1, dtype=np.int32)

        self.mindeg = n
        for i in range(n):
            if self.var_active[i] == 1:
                d = min(n, max(0, int(self.degree[i])))
                self._bucket_insert_front(i, d)
                if d < self.mindeg:
                    self.mindeg = d

        self._apply_dense_postponement()

    # ---------------- Dense postponement ----------------
    # replace _apply_dense_postponement
    def _apply_dense_postponement(self):
        n = self.n
        if self.dense_cutoff is None:
            dense_cut = max(16, min(n - 1, int(10.0 * np.sqrt(n))))
        elif self.dense_cutoff <= 0:
            return
        else:
            dense_cut = int(self.dense_cutoff)

        dense_nodes = [i for i in range(n) if self.var_active[i] == 1 and self.degree[i] >= dense_cut]
        if not dense_nodes:
            return
        for v in dense_nodes:
            self._bucket_remove(v)
            self.where[v] = -1
        self.dense_queue = dense_nodes[:]

    # ---------------- Degree buckets ----------------

    def _bucket_insert_front(self, v: int, d: int):
        h = self.head[d]
        self.prev[v] = -1
        self.next[v] = h
        if h != -1:
            self.prev[h] = v
        self.head[d] = v
        self.where[v] = d

    def _bucket_remove(self, v: int):
        d = self.where[v]
        if d == -1:
            return
        pv, nx = self.prev[v], self.next[v]
        if pv != -1:
            self.next[pv] = nx
        else:
            self.head[d] = nx
        if nx != -1:
            self.prev[nx] = pv
        self.prev[v] = -1
        self.next[v] = -1
        self.where[v] = -1

    def _bucket_move(self, v: int, newd: int):
        if self.var_active[v] == 0:
            return
        self._bucket_remove(v)
        newd = min(self.n, max(0, int(newd)))
        self._bucket_insert_front(v, newd)

    # ---------------- Dynamic iw growth / node growth ----------------

    def _grow_iw(self, required_capacity: int):
        new_cap = max(required_capacity, int(max(2 * self.nzmax, self.nzmax + self.nzmax // 2) + 1024))
        iw_new = np.zeros(new_cap, dtype=np.int32)
        iw_new[:self.nzmax] = self.iw[:self.nzmax]
        self.iw = iw_new
        self.nzmax = new_cap

    def _grow_nodes(self):
        new_nsym = int(self.nsym + max(self.n, self.nsym // 2) + 32)
        self.pe = self._grow_like(self.pe, new_nsym, fill=-1)
        self.len = self._grow_like(self.len, new_nsym, fill=0)
        self.elen = self._grow_like(self.elen, new_nsym, fill=-2)
        self.nv = self._grow_like(self.nv, new_nsym, fill=0)
        self.degree = self._grow_like(self.degree, new_nsym, fill=0)
        self.w = self._grow_like(self.w, new_nsym, fill=0)
        self.last = self._grow_like(self.last, new_nsym, fill=-1)
        self.next = self._grow_like(self.next, new_nsym, fill=-1)
        self.prev = self._grow_like(self.prev, new_nsym, fill=-1)
        self.where = self._grow_like(self.where, new_nsym, fill=-1)
        self.var_active = self._grow_like(self.var_active, new_nsym, dtype=np.int8, fill=0)
        self.elem_active = self._grow_like(self.elem_active, new_nsym, dtype=np.int8, fill=0)
        self.nsym = new_nsym

    @staticmethod
    def _grow_like(arr: np.ndarray, new_size: int, dtype=None, fill=0):
        if dtype is None:
            dtype = arr.dtype
        new = np.full(new_size, fill, dtype=dtype)
        new[:len(arr)] = arr
        return new

    # ---------------- Elimination loop ----------------

    def _select_pivot(self) -> int:
        """
        Tie-breaking: among current min-degree bucket, pick the variable with
        larger nv (supervariable size), then smaller id.
        """
        n = self.n
        while self.mindeg <= n and self.head[self.mindeg] == -1:
            self.mindeg += 1
        if self.mindeg > n:
            # Use postponed dense vars if any
            while self.dense_queue:
                v = self.dense_queue.pop(0)
                if self.var_active[v] == 1 and self.nv[v] > 0:
                    return v
            return -1

        best = -1
        best_nv = -1
        best_id = 1 << 30
        v = self.head[self.mindeg]
        while v != -1:
            nxt = self.next[v]
            if self.var_active[v] == 1 and self.elen[v] == -1 and self.nv[v] > 0:
                nv = int(self.nv[v])
                if nv > best_nv or (nv == best_nv and v < best_id):
                    best = v
                    best_nv = nv
                    best_id = v
            v = nxt

        if best != -1:
            self._bucket_remove(best)
            return best

        # If nothing qualified, advance to next degree
        self.mindeg += 1
        return self._select_pivot()

    def _eliminate_all(self):
        n = self.n
        self.order.clear()
        self._in_order[:] = 0

        while True:
            piv = self._select_pivot()
            if piv == -1:
                break

            # 1) Append the *whole supervariable group* of piv to elimination order
            group = self.sv_members[piv]
            if group:
                group_sorted = sorted(group)
                for g in group_sorted:
                    if self._in_order[g] == 0:
                        self.order.append(g)
                        self._in_order[g] = 1

            # 2) Eliminate pivot representative structurally
            self._eliminate_pivot_build_element(piv)

            # 3) Rarely compact iw to keep locality and cap memory
            self._maybe_compact_iw()

        # Safety: append any originals that somehow didn't enter order
        if len(self.order) < n:
            missing = [i for i in range(n) if self._in_order[i] == 0]
            self.order.extend(missing)

        # Final permutation = reverse of elimination order (factorization ordering)
        self.perm = np.array(self.order[::-1], dtype=np.int32)

    # --------- Pivot elimination: element absorption + var coalescence ---------

    def _eliminate_pivot_build_element(self, piv: int):
        if self.var_active[piv] == 0:
            return

        # Mark pivot dead as variable
        self.var_active[piv] = 0
        self.nv[piv] = 0

        # Gather neighbors
        neighbors = self.iw[self.pe[piv]: self.pe[piv] + self.len[piv]]

        var_neighbors: List[int] = []
        elem_neighbors: List[int] = []
        for u in neighbors:
            if u < 0 or u >= self.nsym:
                continue
            if self.elen[u] == -1:
                if self.var_active[u] == 1 and self.nv[u] > 0:
                    var_neighbors.append(u)
            elif self.elen[u] >= 0:
                if self.elem_active[u] == 1:
                    elem_neighbors.append(u)

        # Clean element neighbors (drop piv/dead)
        cleaned_elems = []
        for e in elem_neighbors:
            if self._clean_element_vars_inplace(e, skip_var=piv) > 0:
                cleaned_elems.append(e)
            else:
                self.elem_active[e] = 0
                self.elen[e] = 0

        # Hashed equality/subset absorption among cleaned elements
        if self.aggressive_absorption and cleaned_elems:
            self._absorb_elements_hashed(cleaned_elems)

        # Mark variables in the new element set (union of var_neighbors and elem vars)
        self._bump_wflg()
        wflg = self._wflg
        for v in var_neighbors:
            self.w[v] = wflg
        for e in cleaned_elems:
            if not self.elem_active[e]:
                continue
            pe_e = self.pe[e]
            le_e = self.elen[e]
            for p in range(pe_e, pe_e + le_e):
                v = self.iw[p]
                if 0 <= v < self.n and self.elen[v] == -1 and self.var_active[v] == 1 and self.nv[v] > 0:
                    self.w[v] = wflg

        new_vars = [v for v in range(self.n) if self.w[v] == wflg and self.var_active[v] == 1 and self.nv[v] > 0]

        # Create new element to represent the clique on new_vars
        if new_vars:
            e_new = self._alloc_new_element()
            self._store_element_varlist(e_new, new_vars)
            self.elem_active[e_new] = 1

            # Rebuild adjacency of each v (remove piv, stale; ensure e_new)
            for v in new_vars:
                self._rebuild_var_list_after_fill(v, piv, e_new)

            # Variable supervariable coalescence among new_vars
            self._coalesce_variables_by_element_signature(new_vars)

            # AMD-style approximate degree update (+ occasional refresh) for each v
            self._bump_wflg()
            tag = self._wflg
            for v in new_vars:
                if self.var_active[v] == 1 and self.nv[v] > 0:
                    deg = self._approx_external_degree(v, tag)
                    self.degree[v] = min(self.n, max(0, deg))
                    self._bucket_move(v, self.degree[v])
                    if self.degree[v] < self.mindeg:
                        self.mindeg = self.degree[v]
                    # Rare polish
                    self._maybe_refresh_degree(v, len(self.order))

    # ----- Variable coalescence (identical element sets) -----

    def _coalesce_variables_by_element_signature(self, vlist: List[int]):
        sig_map: Dict[int, List[int]] = {}

        for v in vlist:
            if self.var_active[v] == 0 or self.nv[v] == 0:
                continue
            elems = self._collect_element_neighbors(v)
            sig = self._hash_id_set(elems)
            sig_map.setdefault(sig, []).append(v)

        for _, vars_with_sig in sig_map.items():
            if len(vars_with_sig) < 2:
                continue
            rep = None
            cand_sets: Dict[int, set] = {}
            for v in vars_with_sig:
                if self.var_active[v] == 1 and self.nv[v] > 0:
                    cand_sets[v] = self._collect_element_neighbors(v)
                    if rep is None:
                        rep = v
            if rep is None:
                continue
            rep_set = cand_sets[rep]

            for u in vars_with_sig:
                if u == rep:
                    continue
                if self.var_active[u] == 0 or self.nv[u] == 0:
                    continue
                uset = cand_sets[u]
                if len(uset) == len(rep_set) and uset == rep_set:
                    self._merge_variable_into_rep(rep, u, rep_set)

    def _collect_element_neighbors(self, v: int) -> set:
        s = set()
        start = self.pe[v]
        L = self.len[v]
        for p in range(start, start + L):
            a = self.iw[p]
            if a >= 0 and a < self.nsym and self.elen[a] >= 0 and self.elem_active[a] == 1:
                s.add(int(a))
        return s

    def _merge_variable_into_rep(self, rep: int, u: int, rep_elem_set: set):
        if rep == u or self.var_active[u] == 0 or self.nv[u] == 0:
            return

        # Update nv and deactivate u
        self.nv[rep] += self.nv[u]
        self.nv[u] = 0
        self.var_active[u] = 0

        # Remove u from buckets/dense queue if present
        self._bucket_remove(u)
        if self.dense_queue:
            self.dense_queue = [x for x in self.dense_queue if x != u]

        # Merge group members: ensure we later output ALL originals
        if self.sv_members[u]:
            self.sv_members[rep].extend(self.sv_members[u])
            self.sv_members[u] = []

        # Replace u by rep in each element of rep_elem_set; unique-compact
        for e in rep_elem_set:
            if not self.elem_active[e] or self.elen[e] <= 0:
                continue
            pe_e = self.pe[e]
            le_e = self.elen[e]
            wr = pe_e
            seen_rep = False
            for p in range(pe_e, pe_e + le_e):
                v = self.iw[p]
                if v == u:
                    v = rep
                if v == rep:
                    if seen_rep:
                        continue
                    seen_rep = True
                self.iw[wr] = v
                wr += 1
            self.elen[e] = wr - pe_e
            # Normalize list order for better hashing hits
            if self.elen[e] > 1:
                seg = self.iw[pe_e:pe_e + self.elen[e]]
                seg.sort()
                self.iw[pe_e:pe_e + self.elen[e]] = seg

        # Optional: clear u adjacency
        self.len[u] = 0

    # ----- AMD-style external degree estimate -----

    def _approx_external_degree(self, v: int, tag: int) -> int:
        start = self.pe[v]
        L = self.len[v]
        total_nv = 0

        # Direct variable neighbors
        for p in range(start, start + L):
            a = self.iw[p]
            if a < 0 or a >= self.nsym:
                continue
            if self.elen[a] == -1:
                if self.var_active[a] == 1 and self.nv[a] > 0 and self.w[a] != tag:
                    self.w[a] = tag
                    total_nv += int(self.nv[a])

        # Variables inside adjacent elements
        for p in range(start, start + L):
            a = self.iw[p]
            if a < 0 or a >= self.nsym:
                continue
            if self.elen[a] >= 0 and self.elem_active[a] == 1:
                pe_e = self.pe[a]
                le_e = self.elen[a]
                for q in range(pe_e, pe_e + le_e):
                    u = self.iw[q]
                    if 0 <= u < self.n and self.elen[u] == -1:
                        if self.var_active[u] == 1 and self.nv[u] > 0 and self.w[u] != tag:
                            self.w[u] = tag
                            total_nv += int(self.nv[u])

        if self.w[v] == tag:
            total_nv -= int(max(0, self.nv[v]))
        return max(0, total_nv)

    def _maybe_refresh_degree(self, v: int, iter_k: int):
        # Every 64 pivots, and only if degree is not tiny
        if (iter_k & 63) != 0:
            return
        if self.degree[v] < min(self.n, 8):
            return
        self._bump_wflg()
        tag = self._wflg
        d = self._approx_external_degree(v, tag)
        if d < self.degree[v]:
            self.degree[v] = d
            self._bucket_move(v, d)
            if d < self.mindeg:
                self.mindeg = d

    def _maybe_compact_iw(self):
        # Compact every 256 pivots to keep memory/locality in check
        if (len(self.order) & 255) != 0:
            return
        write = 0
        for i in range(self.nsym):
            if self.elen[i] >= 0 and self.elem_active[i] == 1:
                s, l = self.pe[i], self.elen[i]
                if l > 0:
                    if write + l > self.nzmax:
                        self._grow_iw(write + l)
                    self.iw[write:write + l] = self.iw[s:s + l]
                    self.pe[i] = write
                    write += l
            elif self.elen[i] == -1 and self.var_active[i] == 1:
                s, l = self.pe[i], self.len[i]
                if l > 0:
                    if write + l > self.nzmax:
                        self._grow_iw(write + l)
                    self.iw[write:write + l] = self.iw[s:s + l]
                    self.pe[i] = write
                    write += l
        self._tail_used = write

    def _bump_wflg(self):
        self._wflg += 1
        if self._wflg == 0x7FFFFFF0:
            self._wflg = 1
            self.w.fill(0)

    # ----- Element cleaning & hashed absorption helpers -----

    def _clean_element_vars_inplace(self, e: int, skip_var: int) -> int:
        pe_e = self.pe[e]
        le_e = self.elen[e]
        rd = pe_e
        wr = pe_e
        for p in range(rd, rd + le_e):
            v = self.iw[p]
            if v == skip_var:
                continue
            if v < 0 or v >= self.nsym:
                continue
            if self.elen[v] == -1 and self.var_active[v] == 1 and self.nv[v] > 0:
                if wr >= self.nzmax:
                    self._grow_iw(wr + 1)
                self.iw[wr] = v
                wr += 1
        self.elen[e] = wr - pe_e
        # Normalize order of element variable list for better hashing
        if self.elen[e] > 1:
            seg = self.iw[pe_e:pe_e + self.elen[e]]
            seg.sort()
            self.iw[pe_e:pe_e + self.elen[e]] = seg
        return self.elen[e]
# --- replace inside AMDReorderingArray ---

    @staticmethod
    def _hash_id_set(ids: set) -> int:
        # Pure-Python ints + explicit masking to avoid numpy int32 overflows
        ssum = 0
        xx = 0
        for v in ids:
            v = int(v)
            xx = (xx ^ ((v * 0x9e3779b1) & 0xFFFFFFFF)) & 0xFFFFFFFF
            ssum += v
        key = (int(ssum) + 1315423911 + ((len(ids) & 0xFFFF) << 16) + int(xx)) & 0x7FFFFFFF
        return int(key)


    def _absorb_elements_hashed(self, elems: List[int]):
        # Hash table size ~ power-of-two >= 2*len(elems)
        m = 1
        need = max(4, 2 * len(elems))
        while m < need:
            m <<= 1

        hhead = np.full(m, -1, dtype=np.int32)
        hnext = np.full(self.nsym, -1, dtype=np.int32)

        def e_hash(eid: int) -> int:
            # Pure-Python arithmetic to avoid overflow
            pe_e = int(self.pe[eid])
            le_e = int(self.elen[eid])
            ssum = 0
            xx = 0
            for p in range(pe_e, pe_e + le_e):
                v = int(self.iw[p])
                xx = (xx ^ ((v * 0x9e3779b1) & 0xFFFFFFFF)) & 0xFFFFFFFF
                ssum += v
            key = (int(ssum) + 1315423911 + ((le_e & 0xFFFF) << 16) + int(xx)) & 0x7FFFFFFF
            return int(key & (m - 1))  # m is power-of-two

        for e in elems:
            if not self.elem_active[e] or self.elen[e] <= 0:
                continue
            b = e_hash(e)
            head = hhead[b]

            # mark e's vars with tag_e
            self._bump_wflg()
            tag_e = self._wflg
            pe_e = int(self.pe[e])
            le_e = int(self.elen[e])
            for p in range(pe_e, pe_e + le_e):
                self.w[int(self.iw[p])] = tag_e

            absorbed = False
            j = head
            while j != -1:
                if not self.elem_active[j] or self.elen[j] <= 0:
                    j = hnext[j]
                    continue

                len_e = int(self.elen[e])
                len_j = int(self.elen[j])

                if len_e == len_j:
                    eq = True
                    cnt = 0
                    pj = int(self.pe[j])
                    for q in range(pj, pj + len_j):
                        v = int(self.iw[q])
                        if self.w[v] != tag_e:
                            eq = False
                            break
                        cnt += 1
                    if eq and cnt == len_e:
                        self.elem_active[e] = 0
                        self.elen[e] = 0
                        absorbed = True
                        break

                if len_e < len_j:
                    # j superset of e?
                    self._bump_wflg()
                    tag_j = self._wflg
                    pj = int(self.pe[j])
                    for q in range(pj, pj + len_j):
                        self.w[int(self.iw[q])] = tag_j
                    is_subset = True
                    pe_e = int(self.pe[e])
                    for q in range(pe_e, pe_e + len_e):
                        if self.w[int(self.iw[q])] != tag_j:
                            is_subset = False
                            break
                    if is_subset:
                        self.elem_active[e] = 0
                        self.elen[e] = 0
                        absorbed = True
                        break
                elif len_j < len_e:
                    # e superset of j?
                    is_subset = True
                    pj = int(self.pe[j])
                    for q in range(pj, pj + len_j):
                        if self.w[int(self.iw[q])] != tag_e:
                            is_subset = False
                            break
                    if is_subset:
                        self.elem_active[j] = 0
                        self.elen[j] = 0
                        # keep scanning; e remains

                j = hnext[j]

            if not absorbed:
                hnext[e] = head
                hhead[b] = e
    # ----- Storage / adjacency rebuild helpers -----

    def _alloc_new_element(self) -> int:
        if self.nelem_top >= self.nsym:
            self._grow_nodes()
        e = self.nelem_top
        self.nelem_top += 1
        self.elen[e] = 0
        self.nv[e] = 0
        self.elem_active[e] = 1
        self.var_active[e] = 0
        self.pe[e] = -1
        return e

    def _store_element_varlist(self, e: int, vlist: List[int]):
        # Normalize order for better hashed equality/subset checks
        vlist = sorted(vlist)
        need = len(vlist)
        pos = self._reserve_space(need)
        self.pe[e] = pos
        self.elen[e] = need
        self.len[e] = 0
        self.iw[pos:pos + need] = np.array(vlist, dtype=np.int32)

    def _reserve_space(self, need: int) -> int:
        start = self._tail_used
        end = start + need
        if end > self.nzmax:
            self._grow_iw(end)
        self._tail_used = end
        return start

    def _rebuild_var_list_after_fill(self, v: int, piv: int, new_elem: int):
        start = self.pe[v]
        L = self.len[v]
        rd = start
        wr = start
        seen_elem = False

        for p in range(rd, rd + L):
            a = self.iw[p]
            if a == piv:
                continue
            if a < 0 or a >= self.nsym:
                continue
            if self.elen[a] == -1:
                if self.var_active[a] == 1 and self.nv[a] > 0:
                    if wr >= self.nzmax:
                        self._grow_iw(wr + 1)
                    self.iw[wr] = a
                    wr += 1
            elif self.elen[a] >= 0:
                if self.elem_active[a] == 1:
                    if a == new_elem:
                        seen_elem = True
                    if wr >= self.nzmax:
                        self._grow_iw(wr + 1)
                    self.iw[wr] = a
                    wr += 1

        if self.elem_active[new_elem] == 1 and not seen_elem:
            if wr < start + L:
                if wr >= self.nzmax:
                    self._grow_iw(wr + 1)
                self.iw[wr] = new_elem
                wr += 1
            else:
                seg = self.iw[start:wr].copy()
                new_start = self._reserve_space(len(seg) + 1)
                self.pe[v] = new_start
                self.iw[new_start:new_start + len(seg)] = seg
                self.iw[new_start + len(seg)] = new_elem
                self.len[v] = len(seg) + 1
                return

        self.len[v] = wr - start

    # ----- small helpers -----

    @staticmethod
    def _hash_id_set(ids: set) -> int:
        # Pure-Python ints + explicit masking to avoid numpy int32 overflows
        ssum = 0
        xx = 0
        for v in ids:
            v = int(v)
            xx = (xx ^ ((v * 0x9e3779b1) & 0xFFFFFFFF)) & 0xFFFFFFFF
            ssum += v
        key = (int(ssum) + 1315423911 + ((len(ids) & 0xFFFF) << 16) + int(xx)) & 0x7FFFFFFF
        return int(key)


    # ---------------- Utilities ----------------

    @staticmethod
    def _bandwidth(A: csr_matrix) -> int:
        if A.nnz == 0:
            return 0
        A = A.tocsr()
        bw = 0
        for i in range(A.shape[0]):
            s, e = A.indptr[i], A.indptr[i + 1]
            if s == e:
                continue
            cols = A.indices[s:e]
            if cols.size:
                bw = max(bw, int(np.max(np.abs(cols - i))))
        return bw
        
    def etree_sym(A: csr_matrix) -> np.ndarray:
        """Elimination tree for symmetric A (pattern only). Returns parent array."""
        A = triu(A, k=0, format="csr")
        n = A.shape[0]
        parent = np.full(n, -1, dtype=np.int32)
        ancestor = np.full(n, -1, dtype=np.int32)

        indptr, indices = A.indptr, A.indices
        for j in range(n):
            ancestor[j] = -1
            for p in range(indptr[j], indptr[j+1]):
                i = indices[p]
                if i >= j:  # only rows < j contribute
                    continue
                while i != -1 and i != j:
                    nxt = ancestor[i]
                    ancestor[i] = j
                    if nxt == -1:
                        parent[i] = j
                        i = -1
                    else:
                        i = nxt
        return parent
