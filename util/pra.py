'''
Author: Kachun Lo
Implementation of Personalized Ranking Adaptation(PRA).
    'https://dl.acm.org/citation.cfm?id=3087993.3088031'
Using Popularity Version and mean-std meature.
'''

import random
import numpy as np


def usr_samples(userid, user_items):
    '''
    sample items of # min(len(items), 10)
    '''
    past_items = user_items[userid]
    usr_pra_samples = random.sample(past_items, min(len(past_items), 15))
    return usr_pra_samples


def pra_score(items_list, item_popularity):
    '''
    Input : 
        items_list      : a list of items one of {rec_list, history_sampled_list}
        item_popularity : dict, containing item_popularity
    Output :
        score = pop_mean + pop_std
    '''
#     print('item_popularity[i] : ', [item_popularity[i] if i in item_popularity.keys() else 1 for i in items_list])
    list_pop = [item_popularity[i][0] if len(item_popularity[i])==2 else 1 for i in items_list]  # len==2 : [pop, LongTail(bool)]
#     print('list_pop: ', list_pop)
    pra_score = np.sum(np.mean(list_pop) + np.std(list_pop))
    
    return pra_score


def pra_reranking(full_list, item_idx, top_n, u_pra_samples, item_popularity):
    '''
    Algorithm described in 'https://dl.acm.org/citation.cfm?id=3087993.3088031'
    '''
    conf_setting = {1:  {'max_step': 1 , 'ex_len': 1},
                    5:  {'max_step': 1, 'ex_len': 1},  # ex_len : 5, 10 ,25
                    10: {'max_step': 1, 'ex_len': 1},}  # max_step: 5, 20, 25
    
    conf = conf_setting[top_n]
    
    # Get the top n indices
    arg_index = np.argsort(-full_list)  # sort from Great to Small values

    rec_items = item_idx[arg_index[:top_n]]
    exc_items = item_idx[arg_index[top_n: top_n+conf['ex_len']]]
    
    u_pra = pra_score(u_pra_samples, item_popularity)
    
#     rec_items_list = np.append(rec_items_list, rec_items)  # for computing converage

    for _ in range(conf['max_step']):
        rec_score = pra_score(rec_items, item_popularity)
        e_base = np.abs(rec_score - u_pra)
        e_delta = 0
        
        # in reverse order
        for i in range(len(rec_items)-1, -1, -1):
            for j in range(len(exc_items)):
                temp_rec = rec_items.copy()
                temp_rec[i] = exc_items[j]
                e_temp = np.abs(pra_score(temp_rec, item_popularity) - u_pra)
                
                if (e_base-e_temp) > e_delta:
                    I_best = i
                    J_best = j
                    e_delta = e_base - e_temp
                    
        # greedily swap only ONCE in each step
        if  e_delta > 0:
            rec_items[I_best], exc_items[J_best] = exc_items[J_best], rec_items[I_best]
        else:
            break
    
    # for computing HR and NDCG
    pra_rec_index = [item_idx.tolist().index(item) for item in rec_items]
    
    return rec_items, pra_rec_index
    
    
