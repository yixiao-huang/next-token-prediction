from utils import *
from data_utils import *
from nxt_token_solver import R_solver
def normalize_grad(iter, ITN , paramater_list):
    for p in paramater_list:
        p.grad /= (p.grad.norm() + torch.finfo(p.data.dtype).eps)
        if iter % (ITN / 4) == 0:
            print("Norm of grad: ", p.grad.norm())

def get_loss(token_choice, loss_type, loss, cls_choice, Vocab, out, ni, idx_token, K, model, C_alpha): 
    if loss_type == 'nll':
        nll = torch.nn.NLLLoss()
    elif loss_type == 'ce':
        ce = torch.nn.CrossEntropyLoss()

    if cls_choice == 'iden':
        # if token_choice == 'ortho':
        #     hat_y = (Vocab @ out.squeeze().reshape(ni, -1, 1)).squeeze(-1) # [n, K]
        # else:
        if not torch.is_tensor(idx_token):
            idx_token = torch.tensor(idx_token).unsqueeze(0)
            
        mask = torch.nn.functional.one_hot(idx_token, num_classes=K).float().permute(0, 2, 1).double()
        hat_y = (mask @ model.attn[0].permute(0, 2, 1)).squeeze(-1)

        # Cls = Vocab @ np.linalg.pinv(Vocab.T @ Vocab)
        # hat_y_new = (Cls @ out.permute(0, 2, 1)).squeeze(-1).double()
        # if not torch.allclose(hat_y, hat_y_new):
        #     print("hat_y and hat_y_new are not close")
        #     print(hat_y.squeeze())
        #     print(hat_y_new.squeeze())
        #     raise ValueError("hat_y and hat_y_new are not close")
        # hat_y = hat_y_new
    elif cls_choice == 'gen':
        Cls = torch.clone(Vocab).double()
        hat_y = (Cls @ out.permute(0, 2, 1)).squeeze(-1).double()
    else:
        raise NotImplementedError("Cls choice {} not implemented".format(cls_choice))
    
    # hat_y = (Cls @ out.permute(0, 2, 1)).squeeze(-1).double()
    assert hat_y.shape == (ni, K), "hat_y shape: {}".format(hat_y.shape)
    
    eps = torch.finfo(hat_y.dtype).eps
    if ni == 1:
        C_alpha = C_alpha.unsqueeze(0)
    if loss_type == 'mse':
        pred = hat_y @ Vocab
        target = Vocab[C_alpha].reshape(ni, -1)        
        loss_mse = torch.nn.MSELoss(reduction = 'sum')
        loss += loss_mse(pred, target)
    elif loss_type == 'lsq2':
        Y = torch.nn.functional.one_hot(C_alpha, num_classes = K).double().reshape(ni, 1, -1)
        if cls_choice == 'iden':
            lsq = (1 - hat_y)**2
        elif cls_choice == 'gen':
            lsq = (1 - torch.nn.functional.softmax(hat_y, dim = -1))**2
        # lsq = (1 - torch.nn.functional.softmax(hat_y, dim = -1))**2
        loss += (Y @ lsq.unsqueeze(-1)).squeeze()
        # loss += (1 - Vocab[C_alpha[i]].reshape(1, -1) @ X[i].squeeze().T @ model.attn[0].squeeze())**2
    elif loss_type == 'cal_corr':
        Y = torch.nn.functional.one_hot(C_alpha, num_classes = K).double().reshape(ni, 1, -1)
        loss += (- Y @ hat_y.reshape(ni, -1, 1)).squeeze()
    elif loss_type == 'nll':
        loss += nll(torch.log(hat_y + eps), C_alpha)
    elif loss_type == 'ce':
        loss += ce(hat_y + eps, C_alpha)
    else:
        raise NotImplementedError("Loss type {} not implemented".format(loss_type))
    return hat_y, loss

def train_W(model, X, idx_token, z, token_choice, cls_choice, Vocab, C_alpha, nlayer,
        n, K, di, ei, ITN, toy_case, batch_toy, scc_klst,
        loss_type, optimizer, norm_grad, parameter_list, 
        Wi_list, attn_probs = None, Wi_norm_list= None, factorize_w = False, reg_kq = False):
    
    for it in range(ITN):
        if toy_case and batch_toy:
            loss = torch.zeros(1)
            for i in range(len(X)):
                ni = 1
                out = model(X[i].unsqueeze(0), z[i].unsqueeze(0))
                # hat_y, loss = get_loss(loss_type, loss, cls_choice, Vocab, out, ni, K, C_alpha[i])
                hat_y, loss = get_loss(token_choice, loss_type, loss, cls_choice, Vocab, out, ni, idx_token[i], K, model, C_alpha[i])

                k = idx_token[i][-1]
                # print("Last token: ", k)
                if attn_probs is not None:
                    hatyi = hat_y @ torch.nn.functional.one_hot(torch.tensor(scc_klst[k])).double()
                    attn_probs[di, ei, it] += hatyi.max()
            loss = loss /len(X)
            if factorize_w and reg_kq:
                print("add regularization")
                eps = torch.finfo(hat_y.dtype).eps
                loss += eps * (model.qlist[0].weight.norm()**2+model.klist[0].weight.norm()**2)
            if attn_probs is not None:
                attn_probs[di, ei, it] /= len(X)
        else:
            out = model(X, z)
            loss = 0
            # hat_y, loss = get_loss(loss_type, loss, cls_choice, Vocab, out, n, K, C_alpha)
            hat_y, loss = get_loss(token_choice, loss_type, loss, cls_choice, Vocab, out, n, idx_token, K, model, C_alpha)
            for i in range(n): 
                k = idx_token[i, -1]
                # print("Last token: ", k)
                if attn_probs is not None:
                    hatyi = hat_y[i] @ torch.nn.functional.one_hot(torch.tensor(scc_klst[k])).double()
                    attn_probs[di, ei, it] += hatyi.max()
            # loss += reg_lambda * sum(torch.linalg.norm(p, 2) for p in parameter_list)
            # print("Current iteration: {}, Loss: {:.4f}".format(it, loss))
            loss = loss.mean()
            if reg_kq:
                # print("add regularization")
                eps = torch.finfo(hat_y.dtype).eps
                loss += 0.001 * (model.qlist[0].weight.norm()**2+model.klist[0].weight.norm()**2)
            if attn_probs is not None:
                attn_probs[di, ei, it] /= n
            if np.isnan(loss.item()):
                print("Nan loss")
                raise ValueError("Nan loss")
        use_gd = False
        if not use_gd:
            optimizer.zero_grad()
        # with torch.autograd.detect_anomaly(False):
            # print(loss)
        loss.backward()

        if norm_grad:
            normalize_grad(it, ITN, parameter_list)
        if use_gd:
            for p in parameter_list:
                p.data = p.data - 0.01 * p.grad.data
        else:
            optimizer.step()  
        if use_gd:
            for p in parameter_list:
                p.grad.data.zero_()
        if it % 500 == 0:
            print("Current iteration: {}, Loss: {:.4f}".format(it, loss.item()))
            # print(torch.linalg.norm(model.qklist[0].weight))
        
        for ni in range(nlayer):
            if not factorize_w:
                sol_gd = model.qklist[ni].weight.detach().numpy()
                Wi_norm_list[ni, ei, it] = np.linalg.norm(sol_gd)
                Wi_list[ni, it] = sol_gd
            # corr_list[di, 0, ei, it] = cal_corr(sol_gd, sol_cvx_list[0, 1])
        
    return model

def train_Wfin(model_fin, w_dir, red_X, red_idx_token, z, token_choice, cls_choice, Vocab, C_alpha, nlayer,
               n, K, di, ei, ITN,
               loss_type, optimizer, norm_grad, parameter_list, 
               Wfini_list = None, perp_corr = None, local = True, loss_list = None):
    if local:
        local = " Local "
    else:
        local = " "
    print("Training{}W_fin".format(local))

    for it in range(ITN):
        loss = torch.zeros(1)
        ncnt = 0
        for i in range(len(red_X)):
            ni = 1
            if len(red_X[i]) == 0:
                continue
            ncnt += 1
            out = model_fin(red_X[i], z[i])
            # hat_y, loss = get_loss(loss_type, loss, cls_choice, Vocab, out, ni, K, C_alpha[i])
            if it == 0:
                print(red_idx_token[i])
                print(C_alpha[i])
            hat_y, loss = get_loss(token_choice, loss_type, loss, cls_choice, Vocab, out, ni, red_idx_token[i], K, model_fin, C_alpha[i])
            # print("It {}, example {}, loss, {}".format(it, i, loss))
        loss = loss / n
        # if it % 500 == 0:
        #     print("Training W_fin Current iteration: {}, raw Loss: {:.4f}".format(it, loss.item()))
        
        # loss += reg_wfin * (model_fin.qklist[0].weight.norm()**2)
        if it % 500 == 0:
            print("Training{}W_fin Current iteration: {}, Loss: {:.4f}".format(local, it, loss.item()))
        optimizer.zero_grad()
        loss.backward()
        if loss_list is not None:
            loss_list[di, ei, it] = loss.item()
        if norm_grad:
            normalize_grad(it, ITN, parameter_list)

        optimizer.step()
        if Wfini_list is not None:
            for ni in range(nlayer):
                Wfini_list[ni, it] = np.copy(model_fin.qklist[ni].weight.detach().numpy()) 
    if Wfini_list is not None:    
        print("{}Correlation between W_fin and W_svm: {}".format(local, cal_corr(Wfini_list[0, -1], w_dir)))
    if perp_corr is not None:
        perp_corr[di, ei] = cal_corr(Wfini_list[0, -1], w_dir)
    return model_fin, Wfini_list, perp_corr


def train_W_Wfin(X, idx_token, red_X, red_idx_token, z, 
                token_choice, cls_choice, Vocab, C_alpha, nlayer,
                n, K, di, ei, ITN, 
                model, model_fin, w_dir,
                scc_klst,
                loss_type, optimizer, optimizer_fin, norm_grad, parameter_list, parameter_list_fin, 
                attn_probs, Wdiff_norm_list, corr_list, proj_corr_list, Sfin,
                Wi_norm_list = None, Wfin_norm_list = None,
                local = False):
    for it in range(ITN):
        if it % (ITN / 4) == 0:
            print("Current iteration: {}".format(it))
        out = model(X, z)
        loss = 0
        hat_y, loss = get_loss(token_choice, loss_type, loss, cls_choice, Vocab, out, n, idx_token, K, model, C_alpha)
        # for i in range(n): 
        #     k = idx_token[i, -1]
        #     # print("Last token: ", k)
        #     hatyi = hat_y[i] @ torch.nn.functional.one_hot(torch.tensor(scc_klst[k])).double()
        #     attn_probs[di, ei, it] += hatyi.max()
        loss = loss.mean()
        # attn_probs[di, ei, it] /= n
        optimizer.zero_grad()
        loss.backward()

        if norm_grad:
            normalize_grad(it, ITN, parameter_list)
            
        optimizer.step()  
        if it % (ITN / 4) == 0:
            print("Loss of W: {:.4f}".format(loss.item()))
            # print(torch.linalg.norm(model.qklist[0].weight))
        
        empty_X = True
        for i in range(len(red_X)):
            if len(red_X[i]) != 0:
                empty_X = False
                break
        
        if not empty_X:
            loss = torch.zeros(1)
            ncnt = 0
            if it % (ITN / 4) == 0:
                print("Training Wfin")
            for i in range(len(red_X)):
                ni = 1
                if len(red_X[i]) == 0:
                    continue
                ncnt += 1
                outi = model_fin(red_X[i], z[i])
                # print("Example {}".format(i))
                hat_y, loss = get_loss(token_choice, loss_type, loss, cls_choice, Vocab, outi, ni, red_idx_token[i], K, model_fin, C_alpha[i])
                # print("It {}, example {}, loss, {}".format(it, i, loss))
            loss = loss / n
            if it % (ITN / 4) == 0:
                print("Loss of Wfin: {:.4f}".format(loss.item()))

            optimizer_fin.zero_grad()
            loss.backward()
            if norm_grad:
                normalize_grad(it, ITN, parameter_list_fin)

            optimizer_fin.step()

        for ni in range(nlayer):
            sol_gd = model.qklist[ni].weight.detach().numpy()
            corr_list[di, 0, ei, it] = cal_corr(sol_gd, w_dir)
            prj_w = project_subspace(Sfin, sol_gd)
            if proj_corr_list is not None:
                prj_w_cp = sol_gd - prj_w
                proj_corr_list[di, ei, it] = cal_corr(w_dir, prj_w_cp)
            Wdiff_norm_list[di, ei, it] = np.linalg.norm(prj_w - model_fin.qklist[ni].weight.detach().numpy())
            
            if Wi_norm_list is not None:
                Wi_norm_list[ni, ei, it] = np.linalg.norm(sol_gd)
            if Wfin_norm_list is not None:
                Wfin_norm_list[ni, ei, it] = np.linalg.norm(model_fin.qklist[ni].weight.detach().numpy())
    import cvxpy as cp
    R = cp.Variable(1)
    prob_R = cp.Problem(cp.Minimize(cp.norm(model_fin.qklist[0].weight.detach().numpy() + R * w_dir - model.qklist[0].weight.detach().numpy(), 'fro')))
    prob_R.solve()
    R = R.value
    print("R", R)
    return model, model_fin, R


