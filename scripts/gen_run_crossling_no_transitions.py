import os

langs = ['sv/da', 'bg/ru', 'ru/bg', 'es/pt', 'pt/es']
langs = ['da/sv','ru/bg']
#langs = ['da/sv','ru/bg','ru/uk', 'es/pt']
langs = ['es/pt']
#langs = ['sv','hu','pt','bg']

with open("run_cpu.sh") as f:
  data = f.readlines()

pyscript = "python -u traincrf.py --langs language --tgt_size 100 --batch_size 64 --no_transitions --model_name null_model_dcrf_specific_sum --model_type specific --test" 

for lang in langs:
  wdata = data + [pyscript.replace("language", lang)]
  script_name = "scripts/run_tagger_" + lang.replace("/","-") + "_cross_ling_dcrf_no_transitions-100-test.sh"
  with open(script_name, 'w') as f:
    f.writelines(wdata)
  os.system("sbatch --nodelist=compute-0-11 -J" + lang + "dcrf_cross_ling_no_transitions " + script_name)
