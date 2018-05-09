from __future__ import division, print_function
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import pdb
import numpy as np
import os
import pickle
import utils, models


parser = argparse.ArgumentParser()
parser.add_argument("--treebank_path", type=str, 
                    default="/projects/tir2/users/cmalaviy/ud_exp/ud-treebanks-v2.1/")
parser.add_argument("--optim", type=str, default='adam', choices=["sgd","adam","adagrad","rmsprop"])
parser.add_argument("--emb_dim", type=int, default=128)
parser.add_argument("--hidden_dim", type=int, default=256)
parser.add_argument("--mlp_dim", type=int, default=128)
parser.add_argument("--n_layers", type=int, default=2)
parser.add_argument("--dropout", type=float, default=0.2)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--langs", type=str, default="uk", 
help="Languages separated by delimiter '/' with last language being target language")
parser.add_argument("--tgt_size", type=int, default=None)
parser.add_argument("--model_name", type=str, default="model_pos")
parser.add_argument("--continue_train", action='store_true')
parser.add_argument("--model_type", type=str, default="baseline", choices=["universal","joint","mono","specific","baseline"])
parser.add_argument("--sum_word_char", action='store_true')
parser.add_argument("--sent_attn", action='store_true')
parser.add_argument("--patience", type=int, default=3)
parser.add_argument("--test", action='store_true')
parser.add_argument("--gpu", action='store_true')
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()
print(args)

# Set seed
torch.manual_seed(args.seed)

# Create dictionaries for language codes, morph tags and pos tags
langs = args.langs.split("/")
args.model_name = args.model_type + "".join(["_" + l for l in langs])
if args.sum_word_char:
    args.model_name += "_wc-sum"
if args.sent_attn:
    args.model_name += "_sent-attn"
if args.tgt_size:
    args.model_name += "-" + str(args.tgt_size)

lang_to_code, code_to_lang = utils.get_lang_code_dicts()
print("Reading training data...")

training_data_langwise, train_tgt_labels = utils.read_conll(args.treebank_path, langs, code_to_lang, tgt_size=args.tgt_size, train_or_dev="train")
training_data = []

if args.tgt_size==100 and args.model_type!="mono":
    training_data_langwise[langs[-1]] = training_data_langwise[langs[-1]] * 10
elif args.tgt_size==1000 and args.model_type!="mono":
    training_data_langwise[langs[-1]] = training_data_langwise[langs[-1]]


for l in langs:
    training_data += training_data_langwise[l]

labels_to_ix = train_tgt_labels
# t = str(args.tgt_size) if args.tgt_size is not None else ""
# with open('labels-'+langs[0]+t+'.txt', 'w') as file:
#     file.write(pickle.dumps(labels_to_ix))
# labels_to_ix = dict([(b, a) for a, b in enumerate(train_tgt_labels)])
# labels_to_ix = {v: k for k, v in ix_to_labels.iteritems()}
dev_data_langwise, dev_tgt_labels = utils.read_conll(args.treebank_path, [langs[-1]], code_to_lang, train_or_dev="dev")
dev_data = dev_data_langwise[langs[-1]]

if args.test:
    test_lang = langs[-1]
    test_data_langwise, test_tgt_labels = utils.read_conll(args.treebank_path, [test_lang], code_to_lang, train_or_dev="test", test=True)
    test_data = test_data_langwise[test_lang]

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


if args.model_type=='universal':
    for lang in langs:
        char_to_ix[lang] = len(char_to_ix)

# training_data_langwise.sort(key=lambda x: -len(x[0]))
# test_data.sort(key=lambda x: -len(x[0]))
# train_order = [x*args.batch_size for x in range(int((len(training_data_langwise)-1)/args.batch_size + 1))]
# test_order = [x*args.batch_size for x in range(int((len(test_data)-1)/args.batch_size + 1))]

def main():

    if not os.path.isfile(args.model_name) or args.continue_train:
        if args.continue_train:
            print("Loading tagger model from " + args.model_name + "...")
            tagger_model = torch.load(args.model_name, map_location=lambda storage, loc: storage)
            if args.gpu:
                tagger_model = tagger_model.cuda()

        else:
            tagger_model = models.BiLSTMTagger(args.model_type, args.sum_word_char, word_freq, args.sent_attn, langs, args.emb_dim, args.hidden_dim, 
                            args.mlp_dim, len(char_to_ix), len(word_to_ix), len(labels_to_ix), args.n_layers, args.dropout, args.gpu)
            if args.gpu:
                tagger_model = tagger_model.cuda()

        loss_function = nn.NLLLoss()

        if args.optim=="sgd":
            optimizer = optim.SGD(tagger_model.parameters(), lr=0.1)
        elif args.optim=="adam":
            optimizer = optim.Adam(tagger_model.parameters())
        elif args.optim=="adagrad":
            optimizer = optim.Adagrad(tagger_model.parameters())
        elif args.optim=="rmsprop":
            optimizer = optim.RMSprop(tagger_model.parameters())

        print("Training tagger model...")
        patience_counter = 0
        prev_avg_tok_accuracy = 0
        for epoch in xrange(args.epochs):
            accuracies = []
            sent = 0
            tokens = 0
            cum_loss = 0
            correct = 0
            print("Starting epoch %d .." %epoch)
            for lang in langs:
                lang_id = []
                if args.model_type=="universal":
                    lang_id = [lang]
                for sentence, morph in training_data_langwise[lang]:
                    sent += 1
                    
                    if sent%100==0:
                        
                        print("[Epoch %d] \
                            Sentence %d/%d, \
                            Tokens %d \
                            Cum_Loss: %f \
                            Average Accuracy: %f" 
                            % (epoch, sent, len(training_data), tokens,
                                cum_loss/tokens, correct/tokens))

                    tagger_model.zero_grad()
                    sent_in = []
                    tokens += len(sentence)

                    for word in sentence:
                        s_appended_word  = lang_id + [c for c in word] + lang_id
                        word_in = utils.prepare_sequence(s_appended_word, char_to_ix, args.gpu)
                        # targets = utils.prepare_sequence(s_appended_word[1:], char_to_ix, args.gpu)
                        sent_in.append(word_in)

                    # sent_in = torch.stack(sent_in)
                    tagger_model.char_hidden = tagger_model.init_hidden()
                    tagger_model.hidden = tagger_model.init_hidden()
                
                    targets = utils.prepare_sequence(morph, labels_to_ix, args.gpu)

                    if args.sum_word_char:
                        word_seq = utils.prepare_sequence(sentence, word_to_ix, args.gpu)
                    else:
                        word_seq = None

                    if args.model_type=="specific" or args.model_type=="joint":
                        tag_scores = tagger_model(sent_in, word_idxs=word_seq, lang=lang)
                    else:
                        tag_scores = tagger_model(sent_in, word_idxs=word_seq)

                    values, indices = torch.max(tag_scores, 1)
                    out_tags = indices.cpu().data.numpy().flatten()
                    correct += np.count_nonzero(out_tags==targets.cpu().data.numpy())
                    loss = loss_function(tag_scores, targets)
                    cum_loss += loss.cpu().data[0]
                    loss.backward()
                    optimizer.step()

            print("Loss: %f" % loss.cpu().data.numpy())
            print("Accuracy: %f" %(correct/tokens))
            print("Saving model..")
            torch.save(tagger_model, args.model_name)
            print("Evaluating on dev set...")
            #avg_tok_accuracy, f1_score = eval(tagger_model, curEpoch=epoch)

            # Early Stopping
            #if avg_tok_accuracy <= prev_avg_tok_accuracy:
            #    patience_counter += 1
            #    if patience_counter==args.patience:
            #        print("Model hasn't improved on dev set for %d epochs. Stopping Training." % patience_counter)
            #        break

            #prev_avg_tok_accuracy = avg_tok_accuracy
    else:
        print("Loading tagger model from " + args.model_name + "...")
        tagger_model = torch.load(args.model_name, map_location=lambda storage, loc: storage)
        if args.gpu:
            tagger_model = tagger_model.cuda()

    if args.test:
        avg_tok_accuracy, f1_score = eval(tagger_model, dev_or_test="test")

def eval(tagger_model, curEpoch=None, dev_or_test="dev"):

    eval_data = dev_data if dev_or_test=="dev" else test_data
    correct = 0
    toks = 0
    hypTags = []
    goldTags = []
    all_out_tags = np.array([])
    all_targets = np.array([])
    print("Starting evaluation on %s set... (%d sentences)" % (dev_or_test, len(eval_data)))
    lang_id = []
    if args.model_type=="universal":
        lang_id = [lang]
    s = 0
    for sentence, morph in eval_data:
        tagger_model.zero_grad()
        tagger_model.char_hidden = tagger_model.init_hidden()
        tagger_model.hidden = tagger_model.init_hidden()
        sent_in = []

        for word in sentence:    
            s_appended_word  = lang_id + [c for c in word] + lang_id
            word_in = utils.prepare_sequence(s_appended_word, char_to_ix, args.gpu)
            sent_in.append(word_in)

        targets = utils.prepare_sequence(morph, labels_to_ix, args.gpu)

        if args.sum_word_char:
            word_seq = utils.prepare_sequence(sentence, word_to_ix, args.gpu)
        else:
            word_seq = None

        if args.model_type=="specific":
            tag_scores = tagger_model(sent_in, word_idxs=word_seq, lang=langs[-1], test=True)
        else:
            tag_scores = tagger_model(sent_in, word_idxs=word_seq, test=True)

        values, indices = torch.max(tag_scores, 1)
        out_tags = indices.cpu().data.numpy().flatten()
        hypTags += [labels_to_ix[idx] for idx in out_tags]
        goldTags.append(morph)
        targets = targets.cpu().data.numpy()
        correct += np.count_nonzero(out_tags==targets)
        toks += len(sentence)
        # all_out_tags = np.append(all_out_tags, out_tags)
        # all_targets = np.append(all_targets, targets)
     
    avg_tok_accuracy = correct / toks

    prefix = args.model_type + "_"
    if args.sum_word_char:
        prefix += "_wc-sum"

    if dev_or_test=="dev":
        prefix += "-".join([l for l in langs]) + "_" + dev_or_test + "_" + str(curEpoch)  
    else:
        prefix += "-".join([l for l in langs]) + "_" + dev_or_test

    if args.sent_attn:
        prefix += "-sent_attn"

    if args.tgt_size:
        prefix += "_" + str(args.tgt_size)


    finalTgts = []
    for tags in goldTags:
      for tag in tags:
        finalTgts.append(tag)


    f1_score, f1_micro_score = utils.computeF1(hypTags, finalTgts, prefix, labels_to_ix, baseline=True, write_results=True) 
    #f1_score, f1_micro_score = utils.computeF1(all_out_tags, all_targets, prefix, labels_to_ix, baseline=True, write_results=True)
    print("Test Set Accuracy: %f" % avg_tok_accuracy)
    print("Test Set Avg F1 Score (Macro): %f" % f1_score)
    print("Test Set Avg F1 Score (Micro): %f" % f1_micro_score)

    with open(prefix + '_results_f1.txt', 'a') as file:
        file.write("\nAccuracy: " + str(avg_tok_accuracy) + "\n")

    return avg_tok_accuracy, f1_score

if __name__=="__main__":
    main()


