
Our code is modified based on DNABERT. For more information about DNABERT, please refer to [DNABERT](https://github.com/jerryji1993/DNABERT)

# msBERT-Promoter
A promoter is a specific sequence in DNA that has transcriptional regulatory functions, playing a role in initiating gene expression. Identifying promoters and their strengths can provide valuable information related to human diseases. In recent years, computational methods have gained prominence as an effective means for identifying promoter, offering a more efficient alternative to labor-intensive biological approaches. In this study, a two-stage integrated predictor called "msBERT-Promoter" is proposed for identifying promoters and predicting their strengths. The model incorporates multi-scale sequence information through a tokenization strategy and fine-tunes the DNABERT model. Soft voting is then used to fuse the multi-scale information, effectively addressing the issue of insufficient DNA sequence information extraction in traditional models. To the best of our knowledge, this is the first time an integrated approach has been used in the DNABERT model for promoter identification and strength prediction. Our model achieves accuracy rates of 96.2% for promoter identification and 79.8% for promoter strength prediction, significantly outperforming existing methods. Furthermore, through attention mechanism analysis, we demonstrate that our model can effectively combine local and global sequence information, enhancing its interpretability. This work paves a new path for the application of artificial intelligence in traditional biology.


## Step by step for training model
### 1. Environment setup
### 1.1 Create and activate a new virtual environment
```
conda create -n dnabert python=3.6
conda activate dnabert
```
### 1.2 Install the package and other requirements

(Required)

```
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch

git clone https://github.com/jerryji1993/DNABERT
cd DNABERT
python3 -m pip install --editable .
cd examples
python3 -m pip install -r requirements.txt
```

### 2 Download pre-trained DNABERT

[DNABERT3](https://drive.google.com/file/d/1nVBaIoiJpnwQxiz4dSq6Sv9kBKfXhZuM/view?usp=sharing)

[DNABERT4](https://drive.google.com/file/d/1V7CChcC6KgdJ7Gwdyn73OS6dZR_J-Lrs/view?usp=sharing)

[DNABERT5](https://drive.google.com/file/d/1KMqgXYCzrrYD1qxdyNWnmUYPtrhQqRBM/view?usp=sharing)

[DNABERT6](https://drive.google.com/file/d/1BJjqb5Dl2lNMg2warsFQ0-Xvn1xxfFXC/view?usp=sharing)


### 3 Fine-tune with pre-trained model
"promoter2non_promoter" refers to the first stage, while "strong2weak_promoter" 
refers to the second stage. Please make the necessary changes accordingly.
### 3.1 Finetune command (change KMER=3,4,5,6)

```
export KMER=3
export MODEL_PATH=./ft/promoter2non_promoter/$KMER
export DATA_PATH=sample_data/ft/promoter2non_promoter/$KMER
export OUTPUT_PATH=./ft/promoter2non_promoter/$KMER
python run_finetune.py \
    --model_type dna \
    --tokenizer_name=dna$KMER \
    --model_name_or_path $MODEL_PATH \
    --task_name dnaprom \
    --do_train \
    --do_eval \
    --data_dir $DATA_PATH \
    --max_seq_length 81 \
    --per_gpu_eval_batch_size=32   \
    --per_gpu_train_batch_size=32   \
    --learning_rate 2e-4 \
    --num_train_epochs 5.0 \
    --output_dir $OUTPUT_PATH \
    --evaluate_during_training \
    --logging_steps 100 \
    --save_steps 4000 \
    --warmup_percent 0.1 \
    --hidden_dropout_prob 0.1 \
    --overwrite_output \
    --weight_decay 0.01 \
    --n_process 8
```
### 3.2 Predict command (change KMER=3,4,5,6)

```
export KMER=5
export MODEL_PATH=./ft/promoter2non_promoter/$KMER
export DATA_PATH=sample_data/ft/promoter2non_promoter/$KMER
export PREDICTION_PATH=./result/promoter2non_promoter/$KMER

python run_finetune.py \
    --model_type dna \
    --tokenizer_name=dna$KMER \
    --model_name_or_path $MODEL_PATH \
    --task_name dnaprom \
    --do_predict \
    --data_dir $DATA_PATH  \
    --max_seq_length 81 \
    --per_gpu_pred_batch_size=128   \
    --output_dir $MODEL_PATH \
    --predict_dir $PREDICTION_PATH \
    --n_process 48
```
### 3.3 Ensemble predict command

```
export MODEL_PATH=./ft/promoter2non_promoter/
export DATA_PATH=sample_data/ft/promoter2non_promoter/
export PREDICTION_PATH=./result/promoter2non_promoter/

python run_finetune.py \
    --model_type dna \
    --tokenizer_name=dna$KMER \
    --model_name_or_path $MODEL_PATH \
    --task_name dnaprom \
    --do_ensemble_pred\
    --data_dir $DATA_PATH  \
    --max_seq_length 81 \
    --per_gpu_eval_batch_size=128   \
    --output_dir $MODEL_PATH \
    --result_dir $PREDICTION_PATH \
    --n_process 48
```
