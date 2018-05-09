import os

langs = ['sv/da', 'bg/ru', 'ru/bg', 'es/pt', 'pt/es']
langs = ['ru/bg']
#langs = ['fi/hu','da/sv', 'es/pt']
langs = ['da/sv']

with open("run_gpu.sh") as f:
  data = f.readlines()

pyscript = "python traincrf.py --gpu --langs language --tgt_size 100 --batch_size 64 --model_name null_model_dcrf_specific --test --no_pairwise"

for lang in langs:
  wdata = data + [pyscript.replace("language", lang)]
  script_name = "scripts/run_tagger_" + lang.replace("/","-") + "_cross_ling_dcrf_specific_no-pairwise-100.sh"
  with open(script_name, 'w') as f:
    f.writelines(wdata)
  os.system("sbatch -J " + lang + "100_dcrf_cross_ling " + script_name)
