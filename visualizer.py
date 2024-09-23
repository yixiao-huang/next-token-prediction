
import matplotlib.pyplot as plt
import numpy as np 


def show_corr(ITN, dlist, corr_list, wmm = False, local_corr_list = None, show_leg = True, labels = None, para = 'K', interval = None, save_path = None):
    idx = 0
    if wmm: 
        print("Only shows Wmm")
    else:
        print("Shows R * Wmm + Wfin")
    if wmm:
        corr_idx = 0
    else:
        corr_idx = 1
    
    plt.figure(figsize=(8,6))
    THRED = ITN
    corr_list = corr_list
    if labels is None:
        if wmm:
            labels = [
                # r'$W(t)^{onetoken}$', 
                r'$W(\tau)^{gmm}$', 
                r'$W(\tau)^{lmm}$', 
            ]
        else:
            labels = [
                # r'$W(t)^{onetoken}$', 
                r'$W(\tau)^{gmm} * R + W_{cyc}^{g}$', 
                r'$W(\tau)^{lmm} * R + W_{cyc}^{l}$', 
            ]

    for idx in range(len(dlist)):
        if show_leg:
            label=labels[0]
        else:
            label=' ' * 10
        if len(dlist) > 0:
            label = r'${} = {}$'.format(para, dlist[idx])
        if interval is not None:
            mean_values = np.mean(corr_list[idx], axis = 0)[:THRED]
            mean_values = np.mean(mean_values.reshape(-1, interval), axis=1)

            std_values =  np.std(corr_list[idx, corr_idx], axis = 0)[:THRED]
            std_values = np.mean(std_values.reshape(-1, interval), axis=1)
            plt.plot(mean_values, linewidth=3, label=label)
            plt.fill_between(range(len(mean_values)), mean_values-std_values, mean_values+std_values, alpha=0.15, linewidth=0)
        else:
            plt.plot(np.mean(corr_list[idx, corr_idx], axis = 0)[:THRED], linewidth=3, label=label)
            mean, std = np.mean(corr_list[idx, corr_idx], axis = 0)[:THRED], np.std(corr_list[idx, corr_idx], axis = 0)[:THRED]
            # print(mean[-1], std[-1])
            plt.fill_between(range(THRED), mean-std, mean+std, alpha=0.15, linewidth=0)


        if local_corr_list is not None:
            if show_leg:
                label=labels[1]
            plt.plot(np.mean(local_corr_list[idx, corr_idx], axis = 0)[:THRED], linewidth=3, label=label)
            mean, std = np.mean(local_corr_list[idx, corr_idx], axis = 0)[:THRED], np.std(local_corr_list[idx, corr_idx], axis = 0)[:THRED]
            # print(mean[-1], std[-1])
            plt.fill_between(range(THRED), mean-std, mean+std, alpha=0.15, linewidth=0)

        if show_leg:
            plt.legend(fontsize=20, loc = 'lower right')
        else:
            plt.legend(fontsize=30, loc = 'lower right')
    
        plt.xlabel('Iterations', fontsize=25)
        plt.ylabel('Correlation coefficient', fontsize=25)
        plt.xticks(np.arange(0, ITN+10, ITN/4), fontsize=20)
        plt.yticks(fontsize=25)
        plt.ylim([-0.05, 1.05])
    plt.grid()
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()


def show_proj_corr(ITN, dlist, proj_corr_list, local_proj_corr_list = None, show_leg = True, labels = None, interval = 10, save_dir = None):
    idx = 0

    plt.figure(figsize=(8,6))
    THRED = ITN
    if labels is None:
        labels = [
            # r'$W(t)^{onetoken}$', 
            r'$\Pi_{\mathcal{S}^{\perp}_{cyc}}(W_g(\tau))}$', 
            r'$\Pi_{\mathcal{S}^{\perp}_{cyc}}(W_l(\tau))}$', 
        ]

    for idx in range(len(dlist)):
        if show_leg:
            label=labels[0]
        else:
            label=' ' * 10

        plt.plot(np.mean(proj_corr_list[idx], axis = 0)[:THRED], linewidth=3, label = label)
        mean, std = np.mean(proj_corr_list[idx], axis = 0)[:THRED], np.std(proj_corr_list[idx], axis = 0)[:THRED]
        plt.fill_between(range(THRED), mean-std, mean+std, alpha=0.15, linewidth=0)
    
        if local_proj_corr_list is not None:
            if show_leg:
                label=labels[1]
            plt.plot(np.mean(local_proj_corr_list[idx], axis = 0)[:THRED], linewidth=3, label=label)
            mean, std = np.mean(local_proj_corr_list[idx], axis = 0)[:THRED], np.std(local_proj_corr_list[idx], axis = 0)[:THRED]
            plt.fill_between(range(THRED), mean-std, mean+std, alpha=0.15, linewidth=0)

        plt.legend(fontsize=20, loc = 'lower right')
        plt.xlabel('Iterations', fontsize=25)
        plt.ylabel('Correlation coefficient', fontsize=25)
        plt.xticks(np.arange(0, ITN+10, ITN / 4), fontsize=25)
        plt.yticks(fontsize=25)
        plt.ylim([-0.05, 1.05])
    plt.grid()
    plt.tight_layout()
    plt.show()

def show_norm_diff(ITN, dlist, Wdiff_norm_list, local_Wdiff_norm_list= None, use_square = True, show_leg = True, labels = None, save_path = None, leg_loc= 'upper right', ylim = None,  para = 'K', interval = None):
    idx = 0

    plt.figure(figsize=(8,6))
    THRED = ITN
    if labels is None:
        labels = [
            r'$W(t)^{gmm}$', 
            r'$W(t)^{lmm}$', 
        ]
    if use_square:
        Wdiff_norm_list = Wdiff_norm_list ** 2
        if local_Wdiff_norm_list is not None:
            local_Wdiff_norm_list = local_Wdiff_norm_list ** 2
    
    for idx in range(len(dlist)):
        if show_leg:
            label=labels[0]
        else:
            label=' ' * 10
        if len(dlist) > 0 and local_Wdiff_norm_list is None:
            label = r'${} = {}$'.format(para, dlist[idx])
        if interval is not None:
            mean_values = np.mean(Wdiff_norm_list[idx], axis = 0)[:THRED]
            mean_values = np.mean(mean_values.reshape(-1, interval), axis=1)
            original_x_axis = np.arange(0, ITN, interval)[:len(mean_values)]

            plt.plot(original_x_axis, mean_values, linewidth=3, label=label)
            # plt.fill_between(range(len(mean_values)), mean_values-std_values, mean_values+std_values, alpha=0.15, linewidth=0)
        else:
            plt.plot(np.mean(Wdiff_norm_list[idx], axis = 0)[:THRED], linewidth=3, label = label)
        mean, std = np.mean(Wdiff_norm_list[idx], axis = 0)[:THRED], np.std(Wdiff_norm_list[idx], axis = 0)[:THRED]

        if local_Wdiff_norm_list is not None:
            if show_leg:
                label=labels[1] 
            plt.plot(np.mean(local_Wdiff_norm_list[idx], axis = 0)[:THRED], linewidth=3, label = label)
            mean, std = np.mean(local_Wdiff_norm_list[idx], axis = 0)[:THRED], np.std(local_Wdiff_norm_list[idx], axis = 0)[:THRED]
        

        if show_leg:
            if leg_loc is None:
                leg_loc = 'lower right'
            plt.legend(fontsize=20, loc = leg_loc)
        else:
            plt.legend(fontsize=30, loc = 'right')
        plt.xlabel('Iterations', fontsize=25)
        plt.ylabel('Norm difference', fontsize=25)
        if ylim is not None:
            plt.ylim([1e-10, ylim])
        # plt.ylim([1e-10, 1])
        plt.xticks(np.arange(0, ITN+10, ITN / 4), fontsize=25)
        plt.yticks(fontsize=25)

        # plt.ylim([0, 1.05])
    plt.grid()
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()

def show_attn_probs(ITN, attn_probs, ni, lr):
    # set idx=0 for W-parametrization
    plt.figure(figsize=(8,6))

    THRED = ITN
    # for ni in range(nlayer):

    mean, std = attn_probs[0].mean(0)[:THRED], attn_probs[0].std(0)[:THRED]

    plt.plot(attn_probs[0].mean(0)[:THRED], linewidth=3, label=r'$W_{}$'.format(ni + 1))
    plt.fill_between(range(THRED), mean-std, mean+std, alpha=0.15, linewidth=0)

    plt.xlabel('Iterations', fontsize=25)
    plt.ylabel('Softmax probability', fontsize=25)
    plt.xticks(np.arange(0, ITN+10, 1000), fontsize=20)
    plt.yticks(np.arange(0.4, 1.1, 0.2), fontsize=25)
    plt.grid()
    # plt.legend(fontsize=30)
    plt.tight_layout()
    plt.show()

def show_attn_probs_neg(ITN, attn_probs, ni, lr):
    fig = plt.figure(figsize=(8,6))

    THRED = ITN

    plt.plot(1 - (attn_probs[0].mean(0)[:THRED]), label = r"$W_{}$".format(ni + 1), linewidth=3)

    # plt.xticks(range(0, THRED, 1000), fontsize=25)
    # plt.yticks([], fontsize=25)
    plt.yscale("log")
    plt.xticks(np.arange(0, THRED + 15, 1000), fontsize=20)
    plt.xlim([-10, THRED])

    plt.xlabel(r'Iterations (lr: {})'.format(lr), fontsize=20)
    plt.ylabel(r'1 - Attn', fontsize=30)
    # plt.legend(fontsize=30)

    plt.grid()
    plt.tight_layout()
    plt.show()

def show_corr_neg(ITN, dlist, corr_list):
    # set idx=0 for W-parametrization
    # set idx=1 for (K, Q)-parametrization
    idx = 0

    plt.figure(figsize=(8,6))
    THRED = ITN - 10
    labels = [
        # r'$W(t)^{onetoken}$', 
        r'$W(\tau)^{mm}$', 
        r'$W(\tau)$', 
    ]
    # if skip:
    #     alt_corr_list = corr_list[corr_list[:, :, -1] > 1e-6].reshape(nlayer, -1, ITN)
    # else:
    #     alt_corr_list = corr_list[0][corr_list[0, :, -1] > 1e-6].reshape(1, -1, ITN)
    # # if global_converge:
    # #     assert alt_corr_list.shape[1] == num_glob2 * epochs
    for idx in range(len(dlist)):
        for ii in range(corr_list.shape[1]):
            plt.plot(1 - np.nanmean(corr_list[idx, ii], axis = 0)[:THRED], linewidth=3, label=labels[ii])

        plt.legend(fontsize=20)
        plt.yscale('log')
        plt.xlabel('Iterations', fontsize=25)
        plt.ylabel('1 - Correlation coefficient', fontsize=25)
        # plt.ylim([1e-10, 1])
        plt.xticks(np.arange(0, ITN+10, 1000), fontsize=20)
        plt.yticks(fontsize=25)
        # plt.ylim([0, 1.05])
    plt.grid()
    plt.tight_layout()
    plt.show()

def show_wnorm(ITN, nlayer, Wi_norm_list):
    # set idx=0 for W-parametrization
    # set idx=1 for (K, Q)-parametrization
    idx = 0

    plt.figure(figsize=(8,6))
    THRED = ITN - 10

    for idx in range(nlayer):
        plt.plot(Wi_norm_list[idx].mean(0)[:THRED], linewidth=3, label=r'$W_{}$'.format(idx+1))
        mean, std = Wi_norm_list[idx].mean(0)[:THRED], Wi_norm_list[idx].std(0)[:THRED]
        plt.fill_between(range(THRED), mean-std, mean+std, alpha=0.15, linewidth=0)
        plt.legend(fontsize=30)
        plt.xlabel('Iterations', fontsize=25)
        plt.ylabel('Weight norm', fontsize=25)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=25)
    plt.grid()
    plt.tight_layout()
    plt.show()

def show_local_stat(epochs, dlist, cnt_dist):
    raise ValueError("This is wrong, cannot use scc to verify")
    fig = plt.figure(figsize =(8, 6))
    labels = [
        # r'$W(t)^{onetoken}$', 
        r'$W(\tau)^{gmm}$', 
        r'$W(\tau)^{lmm}$', 
    ]
    cnt_dist /= epochs
    x = np.arange(len(dlist))  # the label locations

    # for ni in range(len(nlayer_lst)):
    # nlayer = nlayer_lst[ni]
    # for ri in range(len(model.qklist)):
    # ri = 0 # Only consider the first layer
    width = 0.8 / cnt_dist.shape[-1]
    multiplier = 0

    for i in range(cnt_dist.shape[-1]):
        offset = width * multiplier
        rects = plt.bar(x + offset, cnt_dist[0, i], width, label=labels[i])
        multiplier += 1

    plt.xticks(x + width, dlist)
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    plt.grid(axis = 'y', linestyle = '--')

    fig.supxlabel('d', fontsize = 20)
    fig.supylabel('Probabilities', fontsize = 20)

    plt.legend(loc=(-0.6, 0.8), fontsize = 15)

    # ax.set_ylim(0, 250)

def show_scc_stat(plist, cnt_list, K, para = 'N', log = False):

    plt.figure(figsize=(8,6))

    mean_values_list = np.mean(cnt_list, axis = 1)
    std_dev_list = np.std(cnt_list, axis = 1)

    plt.errorbar(plist, mean_values_list, yerr=std_dev_list, fmt='o-', capsize=5)
    plt.xlabel(r'${}$'.format(para), fontsize=25)
    plt.xticks(plist, fontsize=25)

    plt.ylabel('# of SCCs', fontsize = 25)
    # plt.gca().set_aspect('equal', adjustable='box')
    # plt.title('Average Values of # of SCCs for Different Length of Input Sequences', fontsize=25)
    if log:
        plt.xscale('log')
    plt.ylim([1, K])
    plt.grid()
    plt.tight_layout()
    plt.show()