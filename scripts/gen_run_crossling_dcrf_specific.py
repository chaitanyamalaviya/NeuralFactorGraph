import os
import sys

#langs = ['da/sv', 'sv/da', 'bg/ru', 'ru/bg', 'es/pt', 'pt/es']
with open("scripts/run_gpu.sh") as f:
  data = f.readlines()

lang = str(sys.argv[2])

pyscript = "python -u traincrf.py --treebank_path " + str(sys.argv[1]) + " --langs " + str(sys.argv[2]) + " --batch_size 64 --model_type specific --model_name nfg_model --gpu --test"

wdata = data + [pyscript.replace("language", lang)]
script_name = "scripts/run_tagger_" + lang.replace("/","-") + "_mono_nfg_test.sh"
with open(script_name, 'w') as f:
  f.writelines(wdata)
os.system("sbatch -J " + lang + "-mono-test " + script_name)
