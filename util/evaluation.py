import numpy as np
import tensorflow as tf
from tqdm import tqdm
from util import pra

def get_model_scores(sess, test_data, neighborhood, input_user_handle, input_item_handle,
                     input_neighborhood_handle, input_neighborhood_length_handle,
                     dropout_handle, score_op, max_neighbors, user_items, return_scores=False):
    
    """
    test_data = dict([positive, np.array[negatives]])
    
    The paper uses Leave-one-out Evaluation.
    So the test_data = {userid: (pos_id, [neg_id1, neg_id2, ...]), ...}    
    """
    
    out = ''
    scores = []
    total_items = []
    usr_pra_samples = []
    progress = tqdm(test_data.items(), total=len(test_data),
                    leave=False, desc=u'Evaluate || ')
    for user, (pos, neg) in progress:
        # put the pos_item in the last position !!
        
        samples = pra.usr_samples(user, user_items)
        usr_pra_samples.append(samples)
        
        item_indices = list(neg) + [pos]

        feed = {
            input_user_handle: [user] * (len(neg) + 1),
            input_item_handle: item_indices,
        }

        if neighborhood is not None:

            neighborhoods, neighborhood_length = np.zeros((len(neg) + 1, max_neighbors),
                                                          dtype=np.int32), np.ones(len(neg) + 1, dtype=np.int32)

            for _idx, item in enumerate(item_indices):
                _len = min(len(neighborhood.get(item, [])), max_neighbors)
                if _len > 0:
                    neighborhoods[_idx, :_len] = neighborhood[item][:_len]
                    neighborhood_length[_idx] = _len
                else:
                    neighborhoods[_idx, :1] = user
            feed.update({
                input_neighborhood_handle: neighborhoods,
                input_neighborhood_length_handle: neighborhood_length
            })
        
        # The output score is logits (meaning [-inf, inf])
        score = sess.run(score_op, feed)
        scores.append(score.ravel())
        
        total_items.append(item_indices)
        
        if return_scores:
            s = ' '.join(["{}:{}".format(n, s) for s, n in zip(score.ravel().tolist(), item_indices)])
            out += "{}\t{}\n".format(user, s)
    # total item indices in eval (not batch by batch indices)
    item_indices = total_items
    if return_scores:
        return scores, out, item_indices, usr_pra_samples
    return scores


def evaluate_model(sess, test_data, neighborhood, input_user_handle, input_item_handle,
                   input_neighborhood_handle, input_neighborhood_length_handle,
                   dropout_handle, score_op, max_neighbors, EVAL_AT=[1, 5, 10], 
                   item_popularity=None, item_unpop=None, user_items=None, logdir=None, epoch=None):
    
    # set to output also item_indices
    scores, out, item_indices, usr_pra_samples = get_model_scores(sess, test_data, neighborhood, input_user_handle, input_item_handle,
                                                      input_neighborhood_handle, input_neighborhood_length_handle,
                                                      dropout_handle, score_op, max_neighbors, user_items, return_scores=True)
    
    hrs, ndcgs, longtails, coverages, unpop_scores = [], [], [], [], []
    s = '\nNovelty-Adaptive Weighting\n'
    pra_hrs, pra_ndcgs, pra_longtails, pra_coverages, pra_unpop_scores= [], [], [], [], []
    pra_s = '\nPRA_Re-Ranking\n'
    
    for k in EVAL_AT:
        hr, ndcg, longtail, coverage, unpop_score, pra_hr, pra_ndcg, pra_longtail, pra_coverage, pra_unpop_score = get_eval(scores, len(scores[0]) - 1, k, 
                                                                                                                             item_indices=item_indices, 
                                                                                                                             item_popularity=item_popularity, item_unpop=item_unpop, 
                                                                                                                             usr_pra_samples=usr_pra_samples)
        
        
        s += "{:<9} {:<9.4f}{:<9} {:<9.4f}{:<9} {:<9.4f}{:<9} {:<9.4f}{:<9} {:<9.4f}\n".format('HR@%s'%k, hr, 'NDCG@%s'%k, ndcg, 
                                                                                       'L_Tail@%s'%k, longtail, 'Coverage@%s'%k, coverage,
                                                                                       'Unpop_score@%s'%k, unpop_score)
        
        pra_s += "{:<9} {:<9.4f}{:<9} {:<9.4f}{:<9} {:<9.4f}{:<9} {:<9}{:<9} {:<9.4f}\n".format('HR@%s'%k, pra_hr, 'NDCG@%s'%k, pra_ndcg, 
                                                                                       'L_Tail@%s'%k, pra_longtail, 'Coverage@%s'%k, 'NA',
                                                                                       'Unpop_score@%s'%k, pra_unpop_score)
        
#         s += "{:<14} {:<14.6f}{:<14} {:.6f} LongtailRate:{:.4}\n".format('HR@%s' % k, hr, 'NDCG@%s' % k, ndcg, longtail)
        hrs.append(hr)
        ndcgs.append(ndcg)
        longtails.append(longtail)
        coverages.append(coverage)
        unpop_scores.append(unpop_score)
        pra_hrs.append(pra_hr)
        pra_ndcgs.append(pra_ndcg)
        pra_longtails.append(pra_longtail)
        pra_coverages.append(pra_coverage)
        pra_unpop_scores.append(pra_unpop_score)
    print(s, pra_s)
    
    # Write evaluation
    if logdir is not None:
        with open("{}/eval_results".format(logdir), 'a') as f:
            f.write('========> Evaluated at Epoch {}\n'.format(epoch))
            f.write(s)
            f.write(pra_s+'\n\n')

    return hrs, ndcgs, longtails, coverages, unpop_scores


def get_eval(scores, index, top_n=10, item_indices=None, item_popularity=None, item_unpop=None, usr_pra_samples=None):
    """
    if the last element is the correct one, then index = len(scores[0])-1
    """
    ndcg = 0.0
    hr = 0.0
    longtail = 0.0
    unpop_score = 0.0
    assert len(scores[0]) > index and index >= 0
    
    # For PRA Re-Ranking
    pra_ndcg = 0.0
    pra_hr = 0.0
    pra_longtail = 0.0
    pra_unpop_score = 0.0
    
    
#     print('scores: ', len(scores), len(scores[0]))
    
    ############  No Longtail Popularity ############
    if item_indices is None:
        # score for user_0, user_1, 2...
        for i in range(len(scores)):
            score = scores[i]
            # Get the top n indices
            arg_index = np.argsort(-score)[:top_n]  # sort from Great to Small values
            if index in arg_index:
                # Get the position
                ndcg += np.log(2.0) / np.log(arg_index.tolist().index(index) + 2.0)
                # Increment
                hr += 1.0
        return hr / len(scores), ndcg / len(scores)
                
                
    ############  With Longtail Popularity ############
    else:
        item_indices = np.array(item_indices)
        # for computing Item_Coverage "Item Coverage (IC@k)"
        total_unique_items = np.unique(item_indices)
        rec_items_list = np.array([])
        
        # score for user_0, user_1, 2...
        for i in range(len(scores)):
            # for user_i
            score = scores[i]
            item_idx = item_indices[i]
            # Get the top n indices
            arg_index = np.argsort(-score)[:top_n]  # sort from Great to Small values
            rec_items = item_idx[arg_index]
            rec_items_list = np.append(rec_items_list, rec_items)

            # compute longtail ratio
            for item in rec_items:
                # sanity check. 
                if item_popularity[item]:
                    unpop_score += item_unpop[item]
                    if item_popularity[item][1] == True:
                        longtail += 1.0
                # For items only exist in test_set, also belong to longtail.
                else:
                    unpop_score += np.log(len(scores) / 1)  #(= 8.6) tot_user/1_user
                    longtail += 1.0

    #                  print('item_popularity: ', item_popularity[item])
                # Simply count the #True==Longtail and #True/top_n

            if index in arg_index:
                # Get the position
                ndcg += np.log(2.0) / np.log(arg_index.tolist().index(index) + 2.0)
                # Increment
                hr += 1.0
                
                
            ##################  Get PRA_ReRank rec_list  ##################

            u_pra_samples = usr_pra_samples[i]
            pra_rec_list, pra_rec_index = pra.pra_reranking(score, item_idx, top_n, u_pra_samples, item_popularity)

            # compute longtail ratio
            for item in pra_rec_list:
                # sanity check. 
                if item_popularity[item]:
                    pra_unpop_score += item_unpop[item]
                    if item_popularity[item][1] == True:
                        pra_longtail += 1.0
                # For items only exist in test_set, also belong to longtail.
                else:
                    pra_unpop_score += np.log(len(scores) / 1)  #(= 8.6) tot_user/1_user
                    pra_longtail += 1.0

    #                  print('item_popularity: ', item_popularity[item])
                # Simply count the #True==Longtail and #True/top_n

            if index in pra_rec_index:
                # Get the position
                pra_ndcg += np.log(2.0) / np.log(pra_rec_index.index(index) + 2.0)
                # Increment
                pra_hr += 1.0
            
        ##################  Get PRA_ReRank rec_list  ##################
        coverage = (len(np.unique(rec_items_list))) / len(total_unique_items)
                
        return hr/len(scores), ndcg/len(scores), longtail/(top_n*len(scores)), coverage, unpop_score/(top_n*len(scores)), \
               pra_hr/len(scores), pra_ndcg/len(scores), pra_longtail/(top_n*len(scores)), None, pra_unpop_score/(top_n*len(scores))
            