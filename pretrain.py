import argparse
import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from util.gmf import PairwiseGMF
from util.helper import BaseConfig
from util.data import Dataset

from util.evaluation import evaluate_model, get_eval, get_model_scores

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-g', '--gpu', help='set gpu device number 0-3', type=str, required=True)
parser.add_argument('--iters', help='Max iters', type=int, default=15)
parser.add_argument('-b', '--batch_size', help='Batch Size', type=int, default=128)
parser.add_argument('-e', '--embedding', help='Embedding Size', type=int, default=50)
parser.add_argument('--dataset', help='path to npz file', type=str, required=True)
parser.add_argument('-n', '--neg', help='Negative Samples Count', type=int, default=4)  # neg_ratio
parser.add_argument('--l2', help='l2 Regularization', type=float, default=0.001)
parser.add_argument('-o', '--output', help='save filename for trained embeddings', type=str,
                    required=True)

parser.add_argument('--use_unpop_weight', default=False, type=bool, help='Use unpopularity matching.')
FLAGS = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

class Config(BaseConfig):
    filename = FLAGS.dataset
    embed_size = FLAGS.embedding
    batch_size = FLAGS.batch_size
    l2 = FLAGS.l2
    user_count = -1
    item_count = -1
    optimizer = 'adam'
    neg_count = FLAGS.neg
    learning_rate = 0.001
    use_unpop_weight = FLAGS.use_unpop_weight
    
config = Config()
dataset = Dataset(config.filename, config.use_unpop_weight)
config.item_count = dataset.item_count
config.user_count = dataset.user_count
item_popularity = dataset._item_popularity ####
item_unpop = dataset.item_unpop ####
user_items = dataset.user_items ####
tf.logging.info(msg="\n\n%s\n\n" % config)

model = PairwiseGMF(config)
sv = tf.train.Supervisor(logdir=None, save_model_secs=0, save_summaries_secs=0)
sess = sv.prepare_or_wait_for_session(
    config=tf.ConfigProto(gpu_options=tf.GPUOptions(
        per_process_gpu_memory_fraction=0.1,
        allow_growth=True)))

for i in range(FLAGS.iters):
    if sv.should_stop():
        break
    progress = tqdm(enumerate(dataset.get_data(FLAGS.batch_size, False, FLAGS.neg)),
                    dynamic_ncols=True, total=(dataset.train_size * FLAGS.neg) // FLAGS.batch_size)
    loss = []
    
    # for eval
    item_users_list, input_neighborhoods, input_neighborhood_lengths, dropout, max_neighbors = None, None, None, None, None
    evaluate_model(sess, dataset.test_data, item_users_list, model.input_users, model.input_items,
                   input_neighborhoods, input_neighborhood_lengths,
                   dropout, model.score, max_neighbors, EVAL_AT=[1, 5, 10],
                   item_popularity=item_popularity, item_unpop=item_unpop, user_items=user_items)
    
    ### In GMF, only batch_example outputted from Dataset 
    ###   example (Triplet) : [user_idx, item_idx, neg_item_idx]
    for k, example in progress:
        
        ratings, pos_unpop_mtc, neg_unpop_mtc = example   

        feed = {
            model.input_users: ratings[:, 0],  # user_id
            model.input_items: ratings[:, 1],  # item_id
            model.input_items_negative: ratings[:, 2],  # neg_item_id
            
            model.pos_unpop_mtc: pos_unpop_mtc,
            model.neg_unpop_mtc: neg_unpop_mtc,
        }
        batch_loss, _ = sess.run([model.loss, model.train], feed)
        loss.append(batch_loss)
        progress.set_description(u"[{}] Loss: {:,.4f} » » » » ".format(i, batch_loss))

    print("Epoch {}: Avg Loss/Batch {:<20,.6f}".format(i, np.mean(loss)))
    
###############################################    EVALUATION    ##################################################

EVAL_AT = [1, 5, 10]
hrs, ndcgs, longtails, coverages, unpop_scores = [], [], [], [], []
s = '\n===>   Novelty-Adaptive Weighting\n'
pra_hrs, pra_ndcgs, pra_longtails, pra_coverages, pra_unpop_scores= [], [], [], [], []
pra_s = '\n===>   PRA_Re-Ranking\n'

# for eval
item_users_list, input_neighborhoods, input_neighborhood_lengths, dropout, max_neighbors = None, None, None, None, None 

scores, out, item_indices, usr_pra_samples = get_model_scores(sess, dataset.test_data, item_users_list, model.input_users, model.input_items,
                                                       input_neighborhoods, input_neighborhood_lengths,
                                                       dropout, model.score, max_neighbors, user_items, True)

for k in EVAL_AT:
    hr, ndcg, longtail, coverage, unpop_score, pra_hr, pra_ndcg, pra_longtail, pra_coverage, pra_unpop_score = get_eval(scores, len(scores[0]) - 1, k, 
                                                                                                                 item_indices=item_indices, 
                                                                                                                 item_popularity=item_popularity, 
                                                                                                                 item_unpop=item_unpop,
                                                                                                                 usr_pra_samples=usr_pra_samples)
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

    s += "{:<9} {:<9.4f}{:<9} {:<9.4f}{:<9} {:<9.4f}{:<9} {:<9.4f}{:<9} {:<9.4f}\n".format('HR@%s'%k, hr, 'NDCG@%s'%k, ndcg, 
                                                                                           'L_Tail@%s'%k, longtail, 'Coverage@%s'%k, coverage,
                                                                                           'Unpop_score@%s'%k, unpop_score)
    
    pra_s += "{:<9} {:<9.4f}{:<9} {:<9.4f}{:<9} {:<9.4f}{:<9} {:<9}{:<9} {:<9.4f}\n".format('HR@%s'%k, pra_hr, 'NDCG@%s'%k, pra_ndcg, 
                                                                                            'L_Tail@%s'%k, pra_longtail, 'Coverage@%s'%k, 'NA',
                                                                                            'Unpop_score@%s'%k, pra_unpop_score)

print(s, pra_s)

user_embed, item_embed, v = sess.run([model.user_memory.embeddings, model.item_memory.embeddings, model.v.w])
np.savez(FLAGS.output, user=user_embed, item=item_embed, v=v)
print('Saving to: %s' % FLAGS.output)
sv.request_stop()
