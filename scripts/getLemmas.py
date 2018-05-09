from conllu.parser import parse
import utils

def main():
  lang_to_code, code_to_lang = utils.get_lang_code_dicts()
  annot_sents = read_conll(['ru','bg','da','sv','es','pt','uk'], code_to_lang, train_or_dev="train")
  
def read_conll(langs, code_to_lang, train_or_dev, tgt_size=None, test=False):
  
  """
   Reads conll formatted file
  """

  treebank_path = "/projects/tir2/users/cmalaviy/ud_exp/ud-treebanks-v2.0/"
  test_treebank_path = "/projects/tir2/users/cmalaviy/ud_exp/ud-test-v2.0-conll2017/input/conll17-ud-test-2017-05-09/"
  # treebank_path = "/projects/tir2/users/cmalaviy/ud_exp/ud-treebanks-conll2017/"

  annot_sents = {}
 
  for lang in langs:
    sent_text = []
    lemmas = []

    train = train_or_dev if not test else "test"

    if not test:
      filepath = treebank_path + "UD_" + code_to_lang[lang] + '/' + lang + "-ud-" + train + ".conllu"
    else:
      filepath = test_treebank_path + lang + "-udpipe.conllu"

    with open(filepath) as f:
      data = f.readlines()[:-1]
      data = [line for line in data if line[0]!='#']
      split_data = " ".join(data).split("\n \n")
      ud = [parse(sent)[0] for sent in split_data]

      if langs[-1]==lang and tgt_size:
        tgt_size = min(tgt_size, len(ud))
        ud = ud[:tgt_size]
      for sent in ud:
        for word in sent:
          lemmas.append(word['lemma'] + "\n")
          sent_text.append(word['form'] + "\n")
          #lemmas.append(" ".join([w for w in word['lemma']]).encode('utf8') + "\n")
          #sent_text.append(" ".join([w for w in word['form']]).encode('utf8') + "\n")

    with open("lemma-words/" + lang+ "_words.txt",'w') as f:
	f.writelines(sent_text)
    with open("lemma-words/" + lang+ "_lemmas.txt", 'w') as f:
   	f.writelines(lemmas)

    annot_sents[lang] = [(w, m) for w, m in zip(sent_text, lemmas)]

  return annot_sents


if __name__=="__main__":
    main()
