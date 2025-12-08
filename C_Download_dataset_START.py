# =============================================================================
#                         DOWNLOAD CSV (primeri)
# =============================================================================

#SST-2 dev (binarno):

%run "C_Download_standard_text_datasets.py" --dataset sst2 --split validation --out "data/sst2_dev.csv"

# MRPC dev (spoj rečenica sa [SEP]):

%run "C_Download_standard_text_datasets.py" --dataset mrpc --split validation --out "data/mrpc_dev.csv"

#AG News test (4 klase):

%run "C_Download_standard_text_datasets.py" --dataset ag_news --split test --out "data/agnews_test.csv"

#Zahtev:
    
pip install datasets # (jednokratno). Skript pravi CSV sa kolonama text,label.

# EVALUACIJA

%run "C_Eval_safetensors.py" --dataset "datasets/TEXT/sst2_dev.csv" --text-col text --label-col label --model-dirs "models/ALBERT_sst2" "models/BERT-Tiny_sst2" "models/DistilBERT_sst2" "models/MiniLM_sst2" "models/MobileBERT_sst2" "models/TinyBERT_sst2" --batch-size 32 --max-length 128 --device cuda
  
%run "C_Eval_safetensors.py" --dataset "datasets/TEXT/mrpc_dev.csv" --text-col text --label-col label --model-dirs "models/ALBERT_sst2" "models/BERT-Tiny_sst2" "models/DistilBERT_sst2" "models/MiniLM_sst2" "models/MobileBERT_sst2" "models/TinyBERT_sst2" --batch-size 32 --max-length 128 --device cuda
  
%run "C_Eval_safetensors.py" --dataset "datasets/TEXT/agnews_test.csv" --text-col text --label-col label --model-dirs "models/ALBERT_sst2" "models/BERT-Tiny_sst2" "models/DistilBERT_sst2" "models/MiniLM_sst2" "models/MobileBERT_sst2" "models/TinyBERT_sst2" --batch-size 32 --max-length 128 --device cuda