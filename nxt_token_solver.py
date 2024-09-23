import numpy as np
import numpy.linalg as npl
import torch
import torch.nn as nn
from sklearn.svm import LinearSVC
import cvxpy as cp

def W_svm_solver_cvxpy(Vocab, n,  d, idx_z, idx_token, Y, univariate = True, onetoken = False, wfin = False, global_sol = None, adj_mat = None, set_O = None, scc_lst = None, THRED = 1e-3, solver = None, vec_flag = False, C_list = None, E_list = None, verbose = False):
    # minimize \frac{1}{2}p^T @ p <-> min ||p||_2^2
    # subject to G @ p <= -1 
    # ids: k -> \alpha_k
    K, di = Vocab.shape
    if Vocab.dim() == 2:
        K = Vocab.shape[0]
        di = Vocab.shape[1]
    else:
        K = Vocab.shape[1]
        di = Vocab.shape[2]
    W = cp.Variable((di, di), name = 'W')
    W_fin = cp.Variable((di, di), name = 'Wfin')

    if univariate:
        objective = cp.Minimize(cp.norm(W, p = 'fro'))
        objective_fin = cp.Minimize(cp.norm(W_fin, p = 'fro'))
    else:
        objective = cp.Minimize(cp.norm(W, p = 'nuc'))
        objective_fin = cp.Minimize(cp.norm(W_fin, p = 'nuc'))

    constraints = []
    constraints_fin = []
    # if separa:
        # for i in range(n):
        #     for j in range(K):
        #         if j == C[i, idx_z[i]].item():
        #             continue
        #         constraints += [(Vocab[j] - Vocab[C[i, idx_z[i]]]).reshape(1, -1) @ W @ Vocab[idx_z[i]].reshape(-1, 1) <= -1]
    # if onetoken:
    #     assert global_sol is not None
    #     idx_y = np.argmax(global_sol, axis = 1)
    #     for i in range(n):
    #         # print(idx_z[i], idx_y[i])
    #         for j in range(K):
    #             if j == idx_y[i]:
    #                 continue
    #             # print("i, j, idx_y[i]", i, j, idx_y[i])

    #             constraints += [(Vocab[j] - Vocab[idx_y[i]]).reshape(1, -1) @ W @ Vocab[idx_z[i]].reshape(-1, 1) <= -1]
    # elif wfin:
    #     assert global_sol is not None
    #     if set_O is None:
    #         set_idx = np.where(global_sol > THRED, np.arange(K) + 0.01, -np.arange(K) - 0.01)
    #         set_O = set_idx[set_idx > 0].reshape(n, -1) 
    #         bar_O = -set_idx[set_idx < 0].reshape(n, -1)
    #         set_O = set_O.astype(int)
    #         bar_O = bar_O.astype(int)
                    
    #     for i in range(n):
    #         # print(set_O[i], bar_O[i])
    #         for j in range(len(set_O[i])):
    #             idx_j = set_O[i][j]
    #             for k in bar_O[i]:
    #                 # print("i, j, k >=1", i, idx_j, k)
    #                 constraints += [(Vocab[k] - Vocab[idx_j]).reshape(1, -1) @ W @ Vocab[idx_z[i]].reshape(-1, 1) <= -1]
    #             for k in range(j + 1, len(set_O[i])):
    #                 idx_k = set_O[i][k]
    #                 # print("i, j, k ==0", i, idx_j, idx_k)
    #                 constraints += [(Vocab[idx_k] - Vocab[idx_j]).reshape(1, -1) @ W @ Vocab[idx_z[i]].reshape(-1, 1) == 0]
    #                 # print(np.log(global_sol[i][idx_k] / global_sol[i][idx_j]))
                    
    #                 constraints_fin += [(Vocab[idx_k] - Vocab[idx_j]).reshape(1, -1) @ W_fin @ Vocab[idx_z[i]].reshape(-1, 1) == np.log(global_sol[i][idx_k] / global_sol[i][idx_j])]

    zero_const = [[] for _ in range(K)]
    neg_const = [[] for _ in range(K)]
    check_dict = {}
    for i in range(n):
        k = idx_token[i][-1].item()
        # if C_list is not None and E_list is not None:
        #     for ci in range(len(C_list[i])):
        #         c = C_list[i][ci].item()
        #         for ji in range(len(idx_token[i])):
        #             j = idx_token[i][ji].item()
        #             if j == c or j in E_list[i]:
        #                 continue
        #             if (j, c, k) in check_dict or (c, j, k) in check_dict:
        #                 continue
        #             if j in C_list[i]:
        #                 zero_const[k].append(Vocab[j] - Vocab[c])
        #                 check_dict[(j, c, k)] = 1
        #             else:
        #                 neg_const[k].append(Vocab[j] - Vocab[c])
        #                 check_dict[(j, c, k)] = 1
        # else:
        if type(Y[i]) == list:
            y_lst = Y[i]
        elif Y[i].dim() == 0:
            y_lst = [Y[i].item()]
        else:
            raise ValueError("Y[i] should be a list or a tensor scalar")
        for yi in y_lst:
            # In local convergence, we may have multiple labels
            # yi = Y[i]
            for ji in range(len(idx_token[i])):
                j = idx_token[i][ji]
                if j == yi:
                    continue
                # if check_dict.get((yi.item(), j.item(), k.item())) is None and check_dict.get((j.item(), yi.item(), k.item())) is None:

                if scc_lst[k][yi] == scc_lst[k][j]:
                    # print("i j k ==0", yi.item(), j.item(), k.item())
                    if not vec_flag:
                        zero_const[k].append(Vocab[yi] - Vocab[j])
                    else:
                    # constraints += [(Vocab[yi] - Vocab[j]).reshape(1, -1) @ W @ Vocab[k].reshape(-1, 1) == 0]
                        constraints += [cp.kron((Vocab[yi] - Vocab[j]).reshape(-1,1), Vocab[k].reshape(-1,1)).T @ cp.vec(W) == 0]
                else:
                    if not vec_flag:
                        neg_const[k].append(Vocab[j] - Vocab[yi])
                    else:
                    # print("i j k >=1", yi.item(), j.item(), k.item())
                        constraints += [cp.kron((Vocab[j] - Vocab[yi]).reshape(-1,1), Vocab[k].reshape(-1,1)).T @ cp.vec(W) <= -1]
                # constraints += [(Vocab[j] - Vocab[yi]).reshape(1, -1) @ W @ Vocab[k].reshape(-1, 1) <= -1]
    import time 
    st = time.time()
    if not vec_flag:
        # for k in range(K):
        #     if zero_const[k] == []:
        #         continue
        #     stacked_diff = cp.vstack(zero_const[k])
        #     # print(stacked_diff.shape)
        #     lhs = cp.kron(Vocab[k].reshape(1, -1), stacked_diff)
        #     rhs = cp.vec(W)
        #     constraints += [lhs @ rhs == 0]
        #     # constraints += [torch.tensor(zero_const[k]) @ W @ Vocab[k].reshape(-1, 1) == torch.zeros(len(zero_const[k]), 1)]
        # for k in range(K):
        #     if neg_const[k] == []:
        #         continue
        #     stacked_diff = cp.vstack(neg_const[k])
        #     lhs = cp.kron(Vocab[k].reshape(1,-1), stacked_diff)
        #     rhs = cp.vec(W)
        #     constraints += [lhs @ rhs <= -1]
        # constraints += [torch.tensor(neg_const[k]) @ W @ Vocab[k].reshape(-1, 1) <= -torch.ones(len(neg_const[k]), 1)]
        for k in range(K):
            if zero_const[k] == []:
                continue
            constraints += [cp.vstack(zero_const[k]) @ W @ Vocab[k].reshape(-1, 1) == 0]
            # constraints += [torch.tensor(zero_const[k]) @ W @ Vocab[k].reshape(-1, 1) == torch.zeros(len(zero_const[k]), 1)]
        for k in range(K):
            if neg_const[k] == []:
                continue
            # constraints += [torch.tensor(neg_const[k]) @ W @ Vocab[k].reshape(-1, 1) <= -torch.ones(len(neg_const[k]), 1)]
            constraints += [cp.vstack(neg_const[k]) @ W @ Vocab[k].reshape(-1, 1) <= -1]
    print("Time for constraints:", time.time() - st)
    # print("Ground truth")
    # for k in range(K):
    #     if scc_lst[k] == []:
    #         continue
    #     for i in range(K):
    #         for j in range(K):
    #             if i == j:
    #                 continue
    #             if scc_lst[k][i] == scc_lst[k][j]:
    #                 print("i, j, k ==0", i, j, k)
    #                 constraints += [(Vocab[i] - Vocab[j]).reshape(1, -1) @ W @ Vocab[k].reshape(-1, 1) == 0]
    #             elif adj_mat[k, i, j] == 1:
    #                 print("i, j, k >=1", i, j, k)
    #                 constraints += [(Vocab[j] - Vocab[i]).reshape(1, -1) @ W @ Vocab[k].reshape(-1, 1) <= -1]
    print("Number of constraints:", len(constraints))
    prob = cp.Problem(objective, constraints)

    if not prob.is_dcp():
        print("prob is NOT DCP:", prob.is_dcp())
    # print("is lp: ", prob.is_lp())
    print("Solving problems")
    if solver is None:
        solver = cp.SCS
    try:
        prob.solve(solver = solver, verbose = verbose)
    except:
        print("prob is NOT DCP:", prob.is_dcp())
        print("constraints:", constraints)
        # try:
        #     prob.solve(solver = 'MOSEK', verbose = True)
        # except:
        raise Exception("CVXPY Error: " + prob.status)
    
    print("Solved")
    print("First solve time:", prob.solver_stats.solve_time)    
    W = np.array(W.value)


    if prob.status not in ["infeasible", "unbounded"]:
        # Otherwise, problem.value is inf or -inf, respectively.
        sols = {}
        for variable in prob.variables():
            sols[variable.name()] = variable.value
        return W
    else:
        print("prob is NOT DCP:", prob.is_dcp())
        print("constraints:", constraints)
        raise Exception("CVXPY Error: " + prob.status)


def R_solver(Wi_list, Wfini_list, w_dir): 
    R = cp.Variable(1)
    prob_R = cp.Problem(cp.Minimize(cp.norm(Wfini_list[0, -1] + R * w_dir - Wi_list[0, -1], 'fro')))
    prob_R.solve()
    R = R.value
    print("R", R)
    return R


class Tarjan:
    class Edge:
        def __init__(self, to, next):
            self.to = to
            self.next = next
    
    def __init__(self, N):
        self.N = N
        self.edge = []
        self.edgecnt = 0

        self.dfn = [0] * N
        self.low = [0] * N
        self.dfncnt = 0
        self.s = [-1] * (N + 1)
        self.instack = [False] * N
        self.tp = 0

        self.scc = [0] * N
        self.sc = 0
        self.sz = [0] * N

        self.head = [-1] * N
    
    def initialize(self, idx_token, C_alpha, mask = None):
        # C_alpha: [N] 
        check_dict = {}

        for i in range(len(idx_token)):
            if mask is not None and mask[i] == 0:
                continue
            cur_node = C_alpha[i]
            flag_label = False
            for j in range(len(idx_token[i])):
                if cur_node == idx_token[i][j]:
                    flag_label = True
                    break
            if not flag_label:
                continue
            # print(cur_node)
            if self.head[cur_node] == -1:
                self.head[cur_node] = self.edgecnt
                prev_edge = -1
            else:
                prev_edge = self.head[cur_node]
            # print(idx_token.shape[1])
            for j in range(len(idx_token[i])):
                if idx_token[i][j] == cur_node:
                    continue
                # print(idx_token[idi,j])
                self.edge.append(self.Edge(idx_token[i][j], prev_edge))
                prev_edge = self.edgecnt
                self.edgecnt += 1
            
            self.head[cur_node] = prev_edge

    def initialize_local(self, idx_token, C_lst, E_lst, mask):
        # C_lst: [N, N_k]
        for i in range(len(idx_token)):
            if mask[i] == 0:
                continue
            for ci in C_lst[i]:
                flag_label = False
                for j in range(len(idx_token[i])):
                    if ci == idx_token[i][j]:
                        flag_label = True
                        break
                if not flag_label:
                    continue

                if self.head[ci] == -1:
                    self.head[ci] = self.edgecnt
                    prev_edge = -1
                else:
                    prev_edge = self.head[ci]
                # print(idx_token.shape[1])
                for j in range(len(idx_token[i])):
                    if idx_token[i][j] == ci:
                        continue
                    # Exclude nodes with probs: 1e-6 - 1e-3
                    if E_lst is not [] and idx_token[i][j] in E_lst[i]:
                        continue
                        
                    # print(idx_token[idi,j])
                    self.edge.append(self.Edge(idx_token[i][j].item(), prev_edge))
                    prev_edge = self.edgecnt
                    self.edgecnt += 1
                
                self.head[ci] = prev_edge    
    
    def run(self):
        for i in range(self.N):
            if self.dfn[i] == 0:
                self.tarjan(i)

    def tarjan(self, u):
        self.dfncnt = self.dfncnt + 1
        self.low[u] = self.dfn[u] = self.dfncnt
        self.tp = self.tp + 1
        self.s[self.tp] = u

        self.instack[u] = True
        i = self.head[u]
        while i != -1:
            v = self.edge[i].to
            if self.dfn[v] == 0:
                self.tarjan(v)
                self.low[u] = min(self.low[u], self.low[v])
            elif self.instack[v]:
                self.low[u] = min(self.low[u], self.dfn[v])
            i = self.edge[i].next
        
        if self.dfn[u] == self.low[u]:
            while self.s[self.tp] != u:
                self.scc[self.s[self.tp]] = self.sc 
                self.sz[self.sc] += 1
                self.instack[self.s[self.tp]] = False
                self.tp -= 1
            
            self.scc[self.s[self.tp]] = self.sc
            self.sz[self.sc] += 1
            self.instack[self.s[self.tp]] = False
            self.tp -= 1
            self.sc = self.sc + 1
    
    def output(self):
        print("Number of SCC: ", self.sc)
        for sci in range(self.sc):
            print("SCC {} : ".format(sci))
            for i in range(self.N):
                if self.scc[i] == sci:
                    print(i, end = " ")
            print()
            print("size: ", self.sz[sci])
            print()
    
    def get_scc(self):
        return self.scc

def W_fin_svm_solver_cvxpy(Vocab, n, T, d, idx_z, C, global_sol, univariate = True, zero_ids=None):
    # minimize \frac{1}{2}p^T @ p <-> min ||p||_2^2
    # subject to G @ p <= -1 
    # idz_z: k -> \alpha_k
    raise ValueError("Deprecated")
    if Vocab.dim() == 2:
        K = Vocab.shape[0]
    else:
        K = Vocab.shape[1]
    W = cp.Variable((d, d), name = 'Wfin')


    if univariate:
        objective = cp.Minimize(cp.norm(W, p = 'fro'))
    else:
        objective = cp.Minimize(cp.norm(W, p = 'nuc'))

    constraints = []

    for i in range(n):
        for j in range(K):
            if global_sol[i][j] == 0:
                continue
            for k in range(K):
                if global_sol[i][k] == 0 or j == k:
                    continue
                # if i == 0:
                #     print("n, j, k, idx_z:", n, j, k)
                tmp = global_sol[i][j] / global_sol[i][k]
                
                if i == 0:
                    print("ratio:", tmp)
                constraints += [(Vocab[j] - Vocab[k]).reshape(1, -1) @ W @ Vocab[idx_z[i]].reshape(-1, 1) == np.log(tmp)]

    prob = cp.Problem(objective, constraints)
    if not prob.is_dcp():
        print("prob is NOT DCP:", prob.is_dcp())
    try:
        prob.solve(requires_grad = False, verbose = False)
    except:
        print("prob is NOT DCP:", prob.is_dcp())
        print("constraints:", constraints)
        raise Exception("CVXPY Error: " + prob.status)
    
    if prob.status not in ["infeasible", "unbounded"]:
        # Otherwise, problem.value is inf or -inf, respectively.
        sols = {}
        for variable in prob.variables():
            sols[variable.name()] = variable.value
        return sols
    else:
        print("prob is NOT DCP:", prob.is_dcp())
        print("constraints:", constraints)
        raise Exception("CVXPY Error: " + prob.status)
