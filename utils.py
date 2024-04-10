import torch
import numpy as np

def normalize_sol(w):
    return w / (np.linalg.norm(w) + 1e-8)
    # return w / (np.linalg.norm(w))

def cal_corr(w1, w2):
    if w1.reshape(-1) @ w2.reshape(-1) == 0:
        return 0
    w1 = normalize_sol(w1)
    w2 = normalize_sol(w2)

    return w1.reshape(-1) @ w2.reshape(-1)

def cal_corr_alt(w1, w2):

    w1 = normalize_sol(w1)
    w2 = normalize_sol(w2)
    return np.trace(w1 @ w2.T)


def normalize_grad(iter, paramater_list):
    for p in paramater_list:
        p.grad /= (p.grad.norm() + torch.finfo(p.data.dtype).eps)
        if iter % 500 == 0:
            print("Norm of grad: ", p.grad.norm())

def projecting_W(w1, w2): 
    # print(w1.shape, w2.shape)
    return w1.reshape(-1) @ w2.reshape(-1) / (w1.reshape(-1) @ w1.reshape(-1)) * w1

def project_subspace(Sfin, W):
    projected_w = np.zeros_like(W)
    for i in range(len(Sfin)):
        projected_w += projecting_W(Sfin[i], W) 
    return projected_w 

def project_subspace_comp(Sfin, W):
    W_Sfin = project_subspace(Sfin, W)
    assert np.isclose(0, cal_corr(W_Sfin, W - W_Sfin))
    return W - W_Sfin


def gs_mat(mat_X):
    # print("Gram-Schmidt")
    Y = []
    for i in range(len(mat_X)):
        temp_mat = mat_X[i]
        for inY in Y:
            proj_mat = projecting_W(inY, mat_X[i])
            # proj_vec = projecting_W_row(inY, mat_X[i])
            #print "i =", i, ", projection vector =", proj_vec
            temp_mat = temp_mat - proj_mat
            #print "i =", i, ", temporary vector =", temp_vec
        # print(temp_vec)
        if np.allclose(temp_mat, np.zeros_like(temp_mat)):
            continue
        # print("norm and corr")
        # print(np.linalg.norm(temp_vec))
        # print(cal_corr(temp_vec, w_dir))
        Y.append(normalize_sol(temp_mat))

    return Y

def cal_Sfin(K, idx_token,red_idx_token, C_alpha, scc_klst, Vocab):
    Sf= []
    for i in range(len(red_idx_token)):
        if len(red_idx_token[i]) == 0:
            continue
        li = idx_token[i][-1]
        yi = C_alpha[i]
        Ci = scc_klst[li][yi]
        # print("Token {} in SCC {}".format(yi, Ci))
        Sc = []
        smean = 0
        scnt = 0
        for k in range(K): 
            if scc_klst[li][k] == Ci:
                Sc.append(np.copy(Vocab[k].detach().numpy()))
                smean += Vocab[k].detach().numpy()
                scnt += 1
        # Sc = Sc - smean / scnt
        Sc = Sc - np.mean(Sc, axis = 0)
        for Sci in Sc:
            tmp = Sci.reshape(-1,1) @ Vocab[li].reshape(1,-1).detach().numpy()
            Sf.append(tmp)
            # Sfin.append(Sci.reshape(-1,1) @ Vocab[li].reshape(1,-1).detach().numpy())
    Sfin = np.array(gs_mat(Sf))
    return Sfin


def collect_converg(separa, model, cvg_thres, cnt_dist, search_key, dict_token, X, di, C_alpha, loss_type):
    if separa:
        attn_prob = model.attn[0][:, 0].detach().numpy() # [n, T]
        pi = (np.mean(np.max(attn_prob, axis = -1), axis = 0) > cvg_thres).sum()
        # if pi < 1:
        #     print("Pi: ", pi)
        #     print(model.attn[0])
        #     print("X: ", X)
        #     print("Y: ", Y)
        #     print("out: ", out)
        #     # raise Exception("Not all samples converge")
        cnt_dist[di, 0] += 1 - pi
        cnt_dist[di, 1] += pi

        # Get the global solution
        idx = np.argmax(attn_prob, axis = -1)
        gd_idx = []
        for i in range(n):
            gd_idx.append(search_key(dict_token, X[i, idx[i]].squeeze()))
        gd_idx = torch.tensor(gd_idx)

        cnt_dist[di, 2] += (gd_idx == C_alpha).all().numpy()
        # tmp = torch.allclose(Y, out)
        
        # num_glob[di] += tmp 
        # if not tmp:
        #     print("Local convergence")
        #     print("Y: ", Y)
        #     print("out: ", out)
        #     # break
    else:
        pass
        # if loss_type == 'ce':
        #     tmp = (torch.abs(global_loss - loss) < 1e-3).numpy() # 1e-2
        #     # if not tmp:
        #     #     print("Local convergence")
        #     #     print("Global loss: ", global_loss)
        #     #     print("Current loss: ", loss)
        #     #     print("X: ", X)
        #     #     print("C_alpha: ", C_alpha)
        #     #     print("idx_token", idx_token)
        #     #     print("hat_y: ", hat_y)
        #     #     # print("Global attn: ", global_sol)
        #     #     # raise Exception("Global loss is not equal to current loss")
        #         # break  
        #     cnt_dist[di, 2] += tmp
        #     cnt_dist[di, 1] += (not tmp)
    return cnt_dist