'''
Author: Kachun Lo
'''

import os
import argparse
from util.helper import get_optimizer_argparse, preprocess_args, create_exp_directory, BaseConfig, get_logging_config
from util.data import Dataset
from util.evaluation import evaluate_model, get_eval, get_model_scores
from util.cmn import CollaborativeMemoryNetwork
import numpy as np
import tensorflow as tf
from logging.config import dictConfig
from tqdm import tqdm

parser = argparse.ArgumentParser(parents=[get_optimizer_argparse()],
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-g', '--gpu', help='set gpu device number 0-3', type=str, required=True)
parser.add_argument('--iters', help='Max iters', type=int, default=30)
parser.add_argument('-b', '--batch_size', help='Batch Size', type=int, default=128)
parser.add_argument('-e', '--embedding', help='Embedding Size', type=int, default=50)
parser.add_argument('--dataset', help='path to file', type=str, required=True)
parser.add_argument('--hops', help='Number of hops/layers', type=int, default=2)
parser.add_argument('-n', '--neg', help='Negative Samples Count', type=int, default=4)
parser.add_argument('--l2', help='l2 Regularization', type=float, default=0.1)
parser.add_argument('-l', '--logdir', help='Set custom name for logdirectory',
                    type=str, default=None)

parser.add_argument('--resume', help='Resume existing from logdir', action="store_true")
parser.add_argument('--pretrain', help='Load pretrained user/item embeddings', type=str,
                    required=True)
parser.set_defaults(optimizer='rmsprop', learning_rate=0.001, decay=0.9, momentum=0.9)

parser.add_argument('--use_unpop_weight', default=False, type=bool, help='Use unpopularity matching.')
FLAGS = parser.parse_args()
preprocess_args(FLAGS)
os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

# Create results in here unless we specify a logdir
BASE_DIR = 'result/'
if FLAGS.logdir is not None and not os.path.exists(FLAGS.logdir):
    os.mkdir(FLAGS.logdir)

class Config(BaseConfig):
    logdir = create_exp_directory(BASE_DIR) if FLAGS.logdir is None else FLAGS.logdir
    filename = FLAGS.dataset
    embed_size = FLAGS.embedding
    batch_size = FLAGS.batch_size
    hops = FLAGS.hops
    l2 = FLAGS.l2
    user_count = -1
    item_count = -1
    optimizer = FLAGS.optimizer
    tol = 1e-5
    neg_count = FLAGS.neg
    optimizer_params = FLAGS.optimizer_params
    grad_clip = 5.0
    decay_rate = 0.9
    learning_rate = FLAGS.learning_rate
    pretrain = FLAGS.pretrain
    max_neighbors = -1
    
    use_unpop_weight = FLAGS.use_unpop_weight
    

config = Config()
print("FLAGS.resume : ", FLAGS.resume)


if FLAGS.resume:
    print('resume\n\n\n\n\n')
    config.save_directory = config.logdir
    config.load()

dictConfig(get_logging_config(config.logdir))
dataset = Dataset(config.filename, config.use_unpop_weight)
config.item_count = dataset.item_count
config.user_count = dataset.user_count
config.save_directory = config.logdir
config.max_neighbors = dataset._max_user_neighbors
item_popularity = dataset._item_popularity ####
item_unpop = dataset.item_unpop ####
user_items = dataset.user_items ####
tf.logging.info("\n\n%s\n\n" % config)

if not FLAGS.resume:
    config.save()

model = CollaborativeMemoryNetwork(config)

sv = tf.train.Supervisor(logdir=config.logdir, save_model_secs=60 * 10,
                         save_summaries_secs=0)

sess = sv.prepare_or_wait_for_session(config=tf.ConfigProto(
    gpu_options=tf.GPUOptions(allow_growth=True)))

if not FLAGS.resume:
    pretrain = np.load(FLAGS.pretrain, allow_pickle=True)
    sess.graph._unsafe_unfinalize()
    tf.logging.info('Loading Pretrained Embeddings.... from %s' % FLAGS.pretrain)
    sess.run([
        model.user_memory.embeddings.assign(pretrain['user']*0.5),
        model.item_memory.embeddings.assign(pretrain['item']*0.5)])

# Train Loop
for i in range(FLAGS.iters):
    if sv.should_stop():
        break

    progress = tqdm(enumerate(dataset.get_data(FLAGS.batch_size, True, FLAGS.neg)),
                    dynamic_ncols=True, total=(dataset.train_size * FLAGS.neg) // FLAGS.batch_size)
    loss = []
    
    # EVAL
    evaluate_model(sess, dataset.test_data, dataset.item_users_list, model.input_users, model.input_items,
                       model.input_neighborhoods, model.input_neighborhood_lengths,
                       model.dropout, model.score, config.max_neighbors, 
                       item_popularity=item_popularity, item_unpop=item_unpop, user_items=user_items, logdir=config.logdir, epoch=i)
    
    for k, example in progress:
        

        ratings, pos_neighborhoods, pos_neighborhood_length, \
        neg_neighborhoods, neg_neighborhood_length, \
        pos_unpop_mtc, neg_unpop_mtc                           = example                   
        

        feed = {
            model.input_users: ratings[:, 0],
            model.input_items: ratings[:, 1],
            model.input_items_negative: ratings[:, 2],
            model.input_neighborhoods: pos_neighborhoods,
            model.input_neighborhood_lengths: pos_neighborhood_length,
            model.input_neighborhoods_negative: neg_neighborhoods,
            model.input_neighborhood_lengths_negative: neg_neighborhood_length,
            
            model.pos_unpop_mtc: pos_unpop_mtc,
            model.neg_unpop_mtc: neg_unpop_mtc,
        }
    
        batch_loss, _ = sess.run([model.loss, model.train], feed)

        loss.append(batch_loss)
        progress.set_description(u"[{}] Loss: {:,.4f} » » » » ".format(i, batch_loss))

    tf.logging.info("Epoch {}: Avg Loss/Batch {:<20,.6f}".format(i, np.mean(loss)))
    
    # EVAL
    if i>9 and i%5==0 :
        evaluate_model(sess, dataset.test_data, dataset.item_users_list, model.input_users, model.input_items,
                       model.input_neighborhoods, model.input_neighborhood_lengths,
                       model.dropout, model.score, config.max_neighbors, 
                       item_popularity=item_popularity, item_unpop=item_unpop, user_items=user_items, logdir=config.logdir, epoch=i)

#EVAL_AT = range(1, 11)
EVAL_AT = [1, 5, 10]
hrs, ndcgs, longtails, coverages, unpop_scores = [], [], [], [], []
s = '\n===>   Novelty-Adaptive Weighting\n'
pra_hrs, pra_ndcgs, pra_longtails, pra_coverages, pra_unpop_scores= [], [], [], [], []
pra_s = '\n===>   PRA_Re-Ranking\n'
scores, out, item_indices, usr_pra_samples = get_model_scores(sess, dataset.test_data, dataset.item_users_list, model.input_users, model.input_items,
                                                       model.input_neighborhoods, model.input_neighborhood_lengths,
                                                       model.dropout, model.score, config.max_neighbors, user_items, True)

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

with open("{}/final_results".format(config.logdir), 'w') as fout:
    header = ','.join([str(k) for k in EVAL_AT])
    fout.write("{},{}\n".format('metric', header))
    ndcg = ','.join([str(x) for x in ndcgs])
    hr = ','.join([str(x) for x in hrs])
    fout.write("ndcg,{}\n".format(ndcg))
    fout.write("hr,{}".format(hr))

tf.logging.info("Saving model...")
# Save before exiting
sv.saver.save(sess, sv.save_path,
              global_step=tf.contrib.framework.get_global_step())
sv.request_stop()
