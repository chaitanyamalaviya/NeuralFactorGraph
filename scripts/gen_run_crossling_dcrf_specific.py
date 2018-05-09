import os

#langs = ['da/sv', 'sv/da', 'bg/ru', 'ru/bg', 'es/pt', 'pt/es']
langs = ['da/sv', 'ru/bg','fi/hu','es/pt']
langs = ['sv', 'bg', 'hu', 'pt']
with open("run_gpu.sh") as f:
  data = f.readlines()

pyscript = "python -u traincrf.py --langs language --batch_size 64 --model_type specific --model_name null_model_dcrf_specific_sum --gpu"

for lang in langs:
  wdata = data + [pyscript.replace("language", lang)]
  script_name = "scripts/run_tagger_" + lang.replace("/","-") + "_cross_ling_dcrf_specificsum.sh"
  with open(script_name, 'w') as f:
    f.writelines(wdata)
  os.system("sbatch -J " + lang + "specific_cross_ling " + script_name)
