import torch 
import math
from nxt_token_solver import Tarjan
import numpy as np



def search_key(dtoken, value):
    for key, val in dtoken.items():
        if (val == value).all():
            return key
    return None

def next_data(n, T, d, K, rep_seq, token_choice, same_last_token, check_asyc, check_label,):
    flag = False
    """
    check_acyc: corresponds to the acyclic assumption
    check_label: corresponds to the realizable label assumption
    
    """
    while (not flag):
        print("Generating data")
        X, Vocab, C, dict_token, idx_token = data_generator(n, T, d, K = K, rep_seq = rep_seq, choice = token_choice, check_label = check_label, same_last_token = same_last_token)
        if check_asyc:
            C_alpha = C
            flag_asyc = True
            scc_klsti = []
            for k in range(K):
                mask = idx_token[:, -1] == k
                cnt = torch.sum(mask).item()
                if cnt == 0:
                    continue
                tarjan_solver = Tarjan(K)
                tarjan_solver.initialize(idx_token[mask], C_alpha[mask])
                tarjan_solver.run()
                scc_lst = tarjan_solver.get_scc()

                if len(set(scc_lst)) < len(scc_lst):
                    # print("Cyclic graph")
                    flag_asyc = False
                    break
                scc_klsti.append(scc_lst)
            if not flag_asyc:
                continue
        else:
            flag_cyc = False
            C_alpha = C
            scc_klsti = []
            for k in range(K):
                mask = idx_token[:, -1] == k
                cnt = torch.sum(mask).item()
                if cnt == 0:
                    continue
                tarjan_solver = Tarjan(K)
                tarjan_solver.initialize(idx_token[mask], C_alpha[mask])
                tarjan_solver.run()
                scc_lst = tarjan_solver.get_scc()

                if len(set(scc_lst)) < len(scc_lst):
                    # print("Cyclic graph")
                    flag_cyc = True
                    break
                scc_klsti.append(scc_lst)
            if not flag_cyc:
                continue
            flag_asyc = True
        if check_label:
            flag_label = False
            # print("Checking the labels")
            cnt = 0
            for i in range(n):
                labeli = C[i].item()
                flagi = False
                for t in range(T):
                    if idx_token[i, t] == labeli:
                        flagi = True
                        break
                if flagi == False:
                    break
                cnt += 1
            if cnt == n:
                flag_label = True
            # print("Each sample contains the label of the last token")
        else:
            flag_label = True
        print("flag_label: ", flag_label)
        print("flag_asyc: ", flag_asyc)
        flag = flag_label and flag_asyc
    return X, Vocab, C, dict_token, idx_token

def data_generator(dim_n, dim_t, dim_d, K = None, rep_seq = True, choice = 'ortho', rho = 0.5, same_last_token = False, check_label = True):
    """
    X: [dim_n, dim_t, dim_d], share the same vocabulary
    Vocab: [K, dim_d]
    choice: token id in the sequence, chosen from [0, K - 1]
    rep_label: allowing different tokens with the same label
    rep_seq: set X to be rank-1 sequence
    choice: token embedding, chosen from ['ortho', 'equi-corr', 'random']

    C: Mapping from token id to label id
    separa: whether allowing same tokens with different labels (different labels will lead to non-separable data)

    """
    if K is None:
        K = dim_d # Setting K equals to d by default
    if choice == 'ortho':

        assert K <= dim_d 
    
        A = torch.randn(dim_d, dim_d).double()

        Vocab = torch.linalg.qr(A)[0][:K] # [dim_n, K, dim_d]
    elif choice == 'equi-corr':
        # Generate a set of K - 1 vectors with correlation rho
        A = torch.randn(dim_d, dim_d).double()
        C = torch.linalg.qr(A)[0][:K] # [dim_n, K, dim_d]
        Vocab = torch.zeros((K - 1, dim_d)).double()
        for i in range(0, K - 1):
            Vocab[i] = math.sqrt(rho) * C[0] + math.sqrt(1 - rho) * C[i + 1]

        for i in range(K - 1):
            assert Vocab[i].norm() - 1 < 1e-5
            for j in range(i + 1, K - 1):
                # Calculate the correlation between Vocab[i] and Vocab[j]
                assert torch.abs(Vocab[i].dot(Vocab[j])) - rho < 1e-5
        K = K - 1
    else:
        # assert dim_t >= K # The sequence must constains all the tokens in the vocabulary
        Vocab = torch.randn(K, dim_d).double()
        Vocab = torch.nn.functional.normalize(Vocab, dim = -1)
    dict_token = {i: Vocab[i] for i in range(K)} # used to find the index of the token
    # Generating labels 
    
    if dim_t <= K:
        if rep_seq:
            choice_0 = torch.multinomial(torch.ones(K).float(), dim_t, replacement = False).expand(dim_n, dim_t)
        else:
            choice_0 = torch.multinomial(torch.ones(K).float().expand(dim_n, K), dim_t, replacement = False)
        choice = choice_0
    else:
        if rep_seq:
            choice_0 = torch.multinomial(torch.ones(K).float(), K, replacement = False).expand(dim_n, K)
            choice_1 = torch.multinomial(torch.ones(K).float(), dim_t - K, replacement = True).expand(dim_n, dim_t - K)
        else:
            choice_0 = torch.multinomial(torch.ones(K).float().expand(dim_n, K), K, replacement = False)
            choice_1 = torch.multinomial(torch.ones(K).float().expand(dim_n, K), dim_t - K, replacement = True)
            # choice_0 = torch.multinomial(torch.ones(K).float(), K, replacement = False).expand(dim_n, dim_t)
        choice = torch.hstack([choice_0, choice_1])
        
    if same_last_token:
        for i in range(dim_n):
            if choice[i, -1] == 0:
                continue
            idx_zero = torch.where(choice[i] == 0)[0]
            if (len(idx_zero) == 0):
                choice[i, -1] = 0
            else:
                tmp = choice[i, -1].item()
                choice[i, -1] = 0
                choice[i, idx_zero[0]] = tmp
    X = torch.stack([Vocab[choice[i]] for i in range(dim_n)])
    
    C = torch.multinomial(torch.ones(K).float(), dim_n, replacement = True) # mapping k -> \alpha_k
    if check_label:
        for i in range(dim_n):
            if C[i] not in choice[i]:
                C[i] = choice[i, torch.randint(len(choice[i]), (1,))]
    return X.double(), Vocab.double(), C, dict_token, choice.clone()

from itertools import chain

def flatten_chain(matrix):
    return list(chain.from_iterable(matrix))

def manual_toy_case(X, Vocab, idx_token, vary_len = False):
    if vary_len:
        # Allowing different lengths 
        idx_token = [
            [1, 0],
            [2, 3, 0],
            [3, 4, 0],
        ]
        X = []
        for i in range(len(idx_token)):
            X.append(Vocab[idx_token[i]])
    else:
        K = 3
        T = K
        # idx_token = torch.tensor([0, 1, 2, 3] * sum(i for i in range(1, 5)) + 
        #      [1, 2, 3, 0] * sum(i for i in range(1, 5)) + 
        #      [3, 0, 1, 2] * sum(i * i for i in range(1, 5)) + 
        #      [2, 3, 0, 1] * sum(i * i for i in range(1, 5))
        #      ).reshape(-1, 4)
        # idx_token = torch.tensor([0, 1, 2] * sum(i for i in range(1, K + 1)) + 
        #      [2, 1, 0] * sum(i for i in range(1, K + 1))
        #      ).reshape(-1, T)
        # idx_token = torch.tensor([0, 1, 2] * sum(i for i in range(1, K + 1)) + 
        #      [2, 1, 0] * sum(i for i in range(1, K + 1)) + 
        #      [2, 0, 1] * sum(i * i for i in range(1, K + 1))
        #      ).reshape(-1, T)
        # idx_token = torch.tensor([0, 1, 2, 3] * sum(i for i in range(1, 5)) + 
        #      [1, 2, 3, 0] * sum(i for i in range(1, 5))
        #      ).reshape(-1, 4)
        idx_token = torch.tensor([
            # [2, 1, 0],
            # [2, 1, 0],
            [0, 1, 2],
            [0, 1, 2],
            [0, 1, 2],
            [0, 1, 2],
            [0, 1, 2],
            [0, 1, 2],

            [2, 1, 0],
            [2, 1, 0],
            [2, 1, 0],
            [2, 1, 0],
            [2, 1, 0],
            [2, 1, 0],
            # [0, 1, 3, 4, 5, 2],
            # # [0, 1, 2],
            # # [0, 1, 2],
            # [1, 2, 3, 4, 5, 0],
            # # [1, 2, 0],
            # # [1, 2, 0],
            # [0, 2, 3, 4, 5, 1],

            # [0, 1, 2, 4, 5, 3],

            # [0, 1, 2, 3, 4, 5],

            # [0, 1, 2, 5, 3, 4],
            # [0, 2, 1], 
            # [0, 2, 1]
            # [3, 2, 4, 0],
            # [1, 3, 5, 0],
            # [2, 1, 3, 0],
            # [2, 5, 1, 0]
            # [1, 2, 0],
        ])
        X = Vocab[idx_token]
    # [n, K]

    # C_lst = [[i] * i for i in range(1, 5)] + [[i] * (5 - i) for i in range(1, 5)]
    # C_lst = [[i] * i for i in range(1, K + 1)] + [[i] * (K + 1 - i) for i in range(1, K + 1)]

    # C_lst = [[i] * i for i in range(1, K + 1)] + [[i] * (K + 1 - i) for i in range(1, K + 1)] + [[i] * (i * i) for i in range(1, K + 1)] + [[i] * ((K + 1 - i) ** 2) for i in range(1, K + 1)]
    # C_lst = [[i] * i for i in range(1, K + 1)] + [[i] * (K + 1 - i) for i in range(1, K + 1)] + [[i] * (i * i) for i in range(1, K + 1)]
    # C = torch.tensor(flatten_chain(C_lst)) - 1
    C = torch.tensor([
        # [1, 0, 0],
        # [2, 0, 0],
        # [3, 0, 0],
        0, 1, 1, 2, 2, 2, 2, 1, 1, 0, 0, 0,
        # 0, 1, 1, 2, 2, 2,
        # 2, 0, 1, 3, 4, 5
        # [2, 0, 1, 0, 2, 0],
        # [5, 0, 1, 0, 0, 0],
        # [3, 0, 1, 0, 0, 0],
        # [5, 0, 1, 0, 0, 0],
    ])
    return X, C, idx_token 


def generate_reduced_ds(X, T, idx_token, C_alpha, scc):
    cyc_X = []
    cyc_idx = []
    for i in range(len(X)):
        flag = False
        yi = idx_token[i][-1]
        for t in range(T):
            if idx_token[i][t] == C_alpha[i]:
                continue
            s1 = scc[yi][C_alpha[i]]
            s2 = scc[yi][idx_token[i][t]]
            if s1 == s2:
                flag = True
                break
        if flag:
            cyc_X.append(X[i].unsqueeze(0))
            cyc_idx.append(idx_token[i].unsqueeze(0))
        else:
            cyc_X.append([])
            cyc_idx.append([])
    

    reduced_X = []
    reduced_idx = []
    for i in range(len(X)):
        if len(cyc_X[i]) == 0:
            reduced_X.append([])
            reduced_idx.append([])
            continue
        mask = torch.zeros(len(X[i])).bool()
        yi = idx_token[i][-1]
        for t in range(T):
            s1 = scc[yi][C_alpha[i]]
            s2 = scc[yi][idx_token[i][t]]
            if s1 == s2:
                mask[t] = True
        # print(X[i][mask].shape)
        Xi = X[i][mask].reshape(-1, X[i].shape[-1])
        
        reduced_X.append(Xi.unsqueeze(0))
        reduced_idx.append(idx_token[i][mask].unsqueeze(0))
    return reduced_X, reduced_idx, cyc_X, cyc_idx
