from __future__ import division, print_function
import argparse
import numpy as np
import pdb
import os
import time
import random

import factorial_crf_tagger
import utils
import unit

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


parser = argparse.ArgumentParser()
parser.add_argument("--treebank_path", type=str, 
                    default="/projects/tir2/users/cmalaviy/ud_exp/ud-treebanks-v2.1/")
parser.add_argument("--optim", type=str, default='adam', choices=["sgd","adam","adagrad"])
parser.add_argument("--lr", type=float, default=0.1)
parser.add_argument("--emb_dim", type=int, default=128)
parser.add_argument("--hidden_dim", type=int, default=256)
parser.add_argument("--mlp_dim", type=int, default=128)
parser.add_argument("--n_layers", type=int, default=2)
parser.add_argument("--dropout", type=float, default=0.2)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--langs", type=str, default="uk",
                    help="Languages separated by delimiter '/' with last language being target language")
parser.add_argument("--tgt_size", type=int, default=None, 
                    help="Number of training sentences for target language")
parser.add_argument("--model_name", type=str, default="model_dcrf")
parser.add_argument("--no_transitions", action='store_true')
parser.add_argument("--no_pairwise", action='store_true')
parser.add_argument("--continue_train", action='store_true')
parser.add_argument("--model_type", type=str, default="baseline", choices=["universal","joint","mono","specific","baseline"])
parser.add_argument("--sum_word_char", action='store_true')
parser.add_argument("--sent_attn", action='store_true')
parser.add_argument("--patience", type=int, default=3)
parser.add_argument("--test", action='store_true')
parser.add_argument("--visualize", action='store_true')
parser.add_argument("--gpu", action='store_true')
parser.add_argument("--unit_test", action='store_true')
parser.add_argument("--unit_test_args", type=str, default="2,2,2")
parser.add_argument("--seed", type=int, default=42)

args = parser.parse_args()
print(args)

# Set seeds
torch.manual_seed(args.seed)
random.seed(args.seed)

langs = args.langs.split("/")
lang_to_code, code_to_lang = utils.get_lang_code_dicts()


# Set model name

args.model_name += "_" + args.model_type + "".join(["_" + l for l in langs])
# if args.sum_word_char:
#     args.model_name += "-wc_sum"
if args.sent_attn:
    args.model_name += "-sent_attn"
if args.tgt_size:
    args.model_name += "-" + str(args.tgt_size)
if args.no_transitions:
    args.model_name += "-no_transitions"
if args.no_pairwise:
    args.model_name += "-no_pairwise"


# Get training data
print("Loading training data...")
training_data_langwise, train_tgt_labels = utils.read_conll(args.treebank_path, langs, code_to_lang, tgt_size=args.tgt_size, train_or_dev="train")
training_data = []
train_lang_ids = []
# labels_to_ix = train_tgt_labels

unique_tags = utils.find_unique_tags(train_tgt_labels, null_label=True)
print("Number of unique tags: %d" % unique_tags.size())
# unique_tags.printTags()

# Oversample target language data
if args.tgt_size==100 and args.model_type!="mono":
    training_data_langwise[langs[-1]] = training_data_langwise[langs[-1]] * 10

# Add null labels to tag sets in training data
training_data_langwise = utils.addNullLabels(training_data_langwise, langs, unique_tags)


# Create batches for training
train_order = []
train_lang_ids = []
startIdx = 0
for l in langs:
    training_data_langwise[l], lang_ids = utils.sortbylength(training_data_langwise[l], [l]*len(training_data_langwise[l]))
    if args.batch_size != 1:
        train_order += utils.get_train_order(training_data_langwise[l], args.batch_size, startIdx=startIdx)
    training_data += training_data_langwise[l]
    train_lang_ids += [l]*len(training_data_langwise[l])
    startIdx = len(training_data)
    

print("%d sentences in training set" %len(training_data))

if args.unit_test:
    training_data = []
    no_tags, no_labels, no_timesteps = [int(arg) for arg in args.unit_test_args.strip().split(",")]
    training_data, train_tgt_labels = unit.create_sample_data(int(no_tags), [int(no_labels)]*int(no_tags), int(no_timesteps))
    # training_data, train_tgt_labels = unit.create_sample_data(int(no_tags), [2,3], int(no_timesteps))
    print(train_tgt_labels)
    print(training_data)
    training_data = [training_data]

dev_data_langwise, dev_tgt_labels = utils.read_conll(args.treebank_path, [langs[-1]], code_to_lang, train_or_dev="dev")
# Add null labels to tag sets in dev data
dev_data_langwise = utils.addNullLabels(dev_data_langwise, [langs[-1]], unique_tags)
dev_data = dev_data_langwise[langs[-1]]
dev_lang_ids = [langs[-1]]*len(dev_data)

## Sort train/valid set before minibatching

dev_data, dev_lang_ids = utils.sortbylength(dev_data, dev_lang_ids)

if args.test:
    test_lang = langs[-1]
    test_data_langwise, test_tgt_labels = utils.read_conll(args.treebank_path, [test_lang], code_to_lang, train_or_dev="test", test=True)
    test_data_langwise = utils.addNullLabels(test_data_langwise, [test_lang], unique_tags)
    test_data = test_data_langwise[test_lang]
    test_data, test_lang_ids = utils.sortbylength(test_data, [langs[-1]]*len(test_data))
    

# Store starting index of each minibatch
if args.batch_size != 1:
    print("Training Set: %d batches" %len(train_order))
    dev_order = utils.get_train_order(dev_data, args.batch_size)
    print("Dev Set: %d batches" %len(dev_order))
    if args.test:
        test_order = utils.get_train_order(test_data, args.batch_size)
        print("Test Set: %d batches" %len(test_order))

else:
    train_order = [(i,i) for i in range(len(training_data))]
    dev_order = [(i,i) for i in range(len(dev_data))]
    if args.test:
        test_order = [(i,i) for i in range(len(test_data))]


# Build word and character dictionaries
word_to_ix = {}
char_to_ix = {}
word_freq = {}
for sent, _ in training_data:
    for word in sent:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
        if word_to_ix[word] not in word_freq:
            word_freq[word_to_ix[word]] = 1
        else:
            word_freq[word_to_ix[word]] += 1 
        for char in word:
            if char not in char_to_ix:
                char_to_ix[char] = len(char_to_ix)

word_to_ix["UNK"] = len(word_to_ix)
char_to_ix["UNK"] = len(char_to_ix)

#@profile
def main():
    if not os.path.isfile(args.model_name) or args.continue_train:
        if args.continue_train:
            print("Loading tagger model from " + args.model_name + "...")
            tagger_model = torch.load(args.model_name, map_location=lambda storage, loc: storage)
            if args.gpu:
                tagger_model = tagger_model.cuda()
        else:
            print("Creating new model...")
            tagger_model = factorial_crf_tagger.DynamicCRF(args, word_freq, langs, len(char_to_ix), \
            										len(word_to_ix), unique_tags)
            if args.gpu:	
                tagger_model = tagger_model.cuda()

        if args.unit_test:
            tests = unit.TestBP()
            labelSum = sum([tag.size() for tag in tagger_model.uniqueTags])
            # Create dummy LSTM features
            lstm_feats = utils.get_var(torch.Tensor(torch.randn(len(training_data[0][0]), labelSum)), args.gpu)
            tests.setUp(tagger_model, training_data[0][1], len(training_data[0][0]), lstm_feats)

        loss_function = nn.NLLLoss()
        # Provide (N,C) log probability values as input 
        # loss_function = nn.CrossEntropyLoss()

        if args.optim=="sgd":
            optimizer = optim.SGD(tagger_model.parameters(), lr=1.0)
        elif args.optim=="adam":
            optimizer = optim.Adam(tagger_model.parameters())
        elif args.optim=="adagrad":
            optimizer = optim.Adagrad(tagger_model.parameters())

        print("Training FCRF-LSTM model...")
        patience_counter = 0
        prev_avg_tok_accuracy = 0
        for epoch in xrange(args.epochs):
            accuracies = []
            sent = 0
            batch_idx = 0
            tokens = 0
            cum_loss = 0
            correct = 0
            random.shuffle(train_order)
            print("Starting epoch %d .." %epoch)

            start_time = time.time()
            for start_idx, end_idx in train_order:
                train_data = training_data[start_idx : end_idx + 1]
                train_sents = [elem[0] for elem in train_data]
                morph_sents = [elem[1] for elem in train_data]

                lang_ids = train_lang_ids[start_idx : end_idx + 1]

                # print(sentence)
                # print(morph)
                sent += end_idx - start_idx + 1
                tokens += sum([len(sentence) for sentence in train_sents])
                batch_idx += 1

                if batch_idx%5==0:
                    print("[Epoch %d] \
                        Sentence %d/%d, \
                        Tokens %d \
                        Cum_Loss: %f \
                        Time: %f \
                        Tokens/Sec: %d"
                        # Average Accuracy: %f"
                        % (epoch, sent, len(training_data), tokens,
                            cum_loss/tokens, time.time() - start_time, tokens/(time.time()-start_time)))
                            # , correct/tokens))

                tagger_model.zero_grad()
                
                sents_in = []

                for i, sentence in enumerate(train_sents):
                    sent_in = []
                    lang_id = []
                    if args.model_type=="universal":
                        lang_id = [lang_ids[i]]

                    for word in sentence:
                        s_appended_word  = lang_id + [c for c in word] + lang_id
                        word_in = utils.prepare_sequence(s_appended_word, char_to_ix, args.gpu)
                        # targets = utils.prepare_sequence(s_appended_word[1:], char_to_ix, args.gpu)
                        sent_in.append(word_in)
                    sents_in.append(sent_in)


                # sents_in = torch.stack(sent_in)
                tagger_model.char_hidden = tagger_model.init_hidden()
                tagger_model.hidden = tagger_model.init_hidden()

                if args.sum_word_char:
                    all_word_seq = []
                    for sentence in train_sents:
                        word_seq = utils.prepare_sequence(sentence, word_to_ix, args.gpu)
                        all_word_seq.append(word_seq)
                else:
                    all_word_seq = None

                if args.model_type=="specific" or args.model_type=="joint":
                    lstm_feat_sents, graph, maxVal = tagger_model(sents_in, morph_sents, word_idxs=all_word_seq, langs=lang_ids)
                else:
                    lstm_feat_sents, graph, maxVal = tagger_model(sents_in, morph_sents, word_idxs=all_word_seq)

                # Skip parameter updates if marginals are not within a threshold
                if maxVal > 10.00:
                    print("Skipping parameter updates...")
                    continue

                # Compute the loss, gradients, and update the parameters     
                all_factors_batch = []

                for k in range(len(train_sents)):
                    all_factors = tagger_model.get_scores(graph, morph_sents[k], lstm_feat_sents[k], k)
                    all_factors_batch.append(all_factors)

                loss = tagger_model.compute_loss(all_factors_batch, loss_function)
                # print("Loss:", loss)

                # values, indices = torch.max(tag_scores, 1)
                # out_tags = indices.cpu().data.numpy().flatten()
                # correct += np.count_nonzero(out_tags==targets.cpu().data.numpy()) 
                # targets = [utils.unfreeze_dict(tags) for tags in morph]
                # correct += utils.getCorrectCount(targets, hypSeq)
                
                cum_loss += loss.cpu().data[0]
                loss.backward()
                # tagger_model.gradient_check(all_factors_batch[0])
                optimizer.step()

            print("Loss: %f" % loss.cpu().data.numpy())
            # print("Accuracy: %f" %(correct/tokens))
            print("Saving model..")
            torch.save(tagger_model, args.model_name)
            if (epoch+1)%4==0:
                print("Evaluating on dev set...")
                avg_tok_accuracy, f1_score = eval_on_dev(tagger_model, curEpoch=epoch)

                # Early Stopping
                if avg_tok_accuracy <= prev_avg_tok_accuracy:
                    patience_counter += 1
                    if patience_counter==args.patience:
                        print("Model hasn't improved on dev set for %d epochs. Stopping Training." % patience_counter)
                        break

                prev_avg_tok_accuracy = avg_tok_accuracy
    else:
        print("Loading tagger model from " + args.model_name + "...")
        tagger_model = torch.load(args.model_name, map_location=lambda storage, loc: storage)
        if args.gpu:
            tagger_model = tagger_model.cuda()
        else:
            tagger_model.gpu = False

        if args.visualize:
            print("[Visualization Mode]")
            utils.plot_heatmap(unique_tags, tagger_model.pairwise_weights, "pair")
            #utils.plot_heatmap(unique_tags, tagger_model.transition_weights, "trans")
            #utils.plot_heatmap(unique_tags, tagger_model.lang_pairwise_weights, "pair", lang_idx=1)
            print("Stored plots in figures/ directory!")

        if args.test:
            avg_tok_accuracy, f1_score = eval_on_dev(tagger_model, dev_or_test="test")



def eval_on_dev(tagger_model, curEpoch=None, dev_or_test="dev"):

    correct = 0
    toks = 0
    all_out_tags = np.array([])
    all_targets = np.array([])

    eval_order = dev_order if dev_or_test=="dev" else test_order
    eval_data = dev_data if dev_or_test=="dev" else test_data

    print("Starting evaluation on %s set... (%d sentences)" % (dev_or_test, len(eval_data)))

    lang_id = []
    if args.model_type=="universal":
        lang_id = [langs[-1]]

    for start_idx, end_idx in eval_order[:1]:

        cur_eval_data = eval_data[start_idx : end_idx + 1]
        eval_sents = [elem[0] for elem in cur_eval_data]
        morph_sents = [elem[1] for elem in cur_eval_data]


        sents_in = []

        for i, sentence in enumerate(eval_sents):
            sent_in = []
            for word in sentence:
                s_appended_word  = lang_id + [c for c in word] + lang_id
                word_in = utils.prepare_sequence(s_appended_word, char_to_ix, args.gpu)
                # targets = utils.prepare_sequence(s_appended_word[1:], char_to_ix, args.gpu)
                sent_in.append(word_in)
            sents_in.append(sent_in)

        tagger_model.zero_grad()
        tagger_model.char_hidden = tagger_model.init_hidden()
        tagger_model.hidden = tagger_model.init_hidden()

        all_word_seq = []
        for sentence in eval_sents:
            word_seq = utils.prepare_sequence(sentence, word_to_ix, args.gpu)
            all_word_seq.append(word_seq)

        if args.model_type=="specific" or args.model_type=="joint":
            lstm_feats, graph, maxVal = tagger_model(sents_in, morph_sents, word_idxs=all_word_seq, langs=[langs[-1]]*len(sents_in), test=True)
        else:
            lstm_feats, graph, maxVal = tagger_model(sents_in, morph_sents, word_idxs=all_word_seq, test=True)

        for k in range(len(eval_sents)):
            hypSeq = tagger_model.getBestSequence(graph, k)
            targets = [utils.unfreeze_dict(tags) for tags in morph_sents[k]]
            #hypSeqTrimmed = []
            #targets = []
            #for i, tags in enumerate(morph_sents[k]):
            #    if utils.removeNullLabels(tags) not in train_tgt_labels:
            #        targets.append(utils.unfreeze_dict(tags))
            #        hypSeqTrimmed.append(hypSeq[i])

            correct += utils.getCorrectCount(targets, hypSeq)
            toks += len(eval_sents[k])
            #toks += len(targets)
            all_out_tags = np.append(all_out_tags, hypSeq)
            all_targets = np.append(all_targets, targets)
    avg_tok_accuracy = correct / toks

    prefix = args.model_name
    prefix += "_" + dev_or_test

    if args.sent_attn:
        prefix += "sent_attn"

    if args.tgt_size:
        prefix += "_" + str(args.tgt_size)

    write = True if dev_or_test=="test" else False

    f1_score, f1_micro_score = utils.computeF1(all_out_tags, all_targets, prefix, write_results=write)
    print("Test Set Accuracy: %f" % avg_tok_accuracy)
    print("Test Set Avg F1 Score (Macro): %f" % f1_score)
    print("Test Set Avg F1 Score (Micro): %f" % f1_micro_score)

    if write:
        with open(prefix + '_results_f1.txt', 'ab') as file:
            file.write("\nAccuracy: " + str(avg_tok_accuracy) + "\n")
            for target, hyp in zip(all_targets, all_out_tags):
                file.write(str(target) + "\n")
                file.write(str(hyp) + "\n")

    return avg_tok_accuracy, f1_score


# def eval_on_test(tagger_model):

#     correct = 0
#     toks = 0
#     all_out_tags = np.array([])
#     all_targets = np.array([])
#     print("Starting evaluation on test set... (%d sentences)" % (len(test_data)))

#     lang_id = []
#     if args.model_type=="universal":
#         lang_id = [lang]
    
#     for sentence, morph in test_data:
#         tagger_model.zero_grad()
#         tagger_model.char_hidden = tagger_model.init_hidden()
#         tagger_model.hidden = tagger_model.init_hidden()
#         sent_in = []

#         for word in sentence:
#             s_appended_word  = lang_id + [c for c in word] + lang_id
#             word_in = utils.prepare_sequence(s_appended_word, char_to_ix, args.gpu)
#             sent_in.append(word_in)
#         # sentence_in = utils.prepare_sequence(sentence, word_to_ix, args.gpu)

#         # targets = utils.prepare_sequence(morph, labels_to_ix, args.gpu)
#         # if args.sum_word_char:
#         word_seq = [utils.prepare_sequence(sentence, word_to_ix, args.gpu)]
#         # else:
#         #     word_seq = None

#         if args.model_type=="specific" or args.model_type=="joint":
#             lstm_feats, graph, maxVal = tagger_model([sent_in], [morph], word_idxs=word_seq, lang=langs[-1], test=True)
#         else:
#             lstm_feats, graph, maxVal = tagger_model([sent_in], [morph], word_idxs=word_seq, test=True)

#         hypSeq = tagger_model.getBestSequence(graph, 0)
#         targets = [utils.unfreeze_dict(tags) for tags in morph]
#         # correct += np.count_nonzero(out_tags==targets)
#         correct += utils.getCorrectCount(targets, hypSeq)
#         toks += len(sentence)
#         all_out_tags = np.append(all_out_tags, hypSeq)
#         all_targets = np.append(all_targets, targets)

#     avg_tok_accuracy = correct / toks

#     prefix = args.model_type + "_"
#     if args.sum_word_char:
#         prefix = "wc-sum-hf_" + prefix

#     prefix += "-".join([l for l in langs]) + "_test" 

#     if args.sent_attn:
#         prefix += "sent_attn"

#     if args.tgt_size:
#         prefix += "_" + str(args.tgt_size)

#     f1_score, f1_micro_score = utils.computeF1(all_out_tags, all_targets, prefix, write_results=True)
#     print("Test Set Accuracy: %f" % avg_tok_accuracy)
#     print("Test Set Avg F1 Score (Macro): %f" % f1_score)
#     print("Test Set Avg F1 Score (Micro): %f" % f1_micro_score)

#     with open(prefix + '_results_f1.txt', 'a') as file:
#         file.write("\nAccuracy: " + str(avg_tok_accuracy) + "\n")

#     return avg_tok_accuracy, f1_score

if __name__=="__main__":
    main()
