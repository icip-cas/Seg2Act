# Seg2Act: Global Context-aware Generation for Document Logical Structuring


## TL;DR

We introduce Seg2Act, a global context-aware action generation approach for document logical structuring.

## News

* [2024/09] Seg2Act is accepted by EMNLP 2024 as a long paper at the main conference!

## Abstract

Document logical structuring aims to extract the underlying hierarchical structure of documents, which is crucial for document intelligence. Traditional approaches often fall short in handling the complexity and the variability of lengthy documents. To address these issues, we introduce Seg2Act, an end-to-end, generation-based method for document logical structuring, revisiting logical structure extraction as an action generation task. Specifically, given the text segments of a document, Seg2Act iteratively generates the action sequence via a global context-aware generative model, and simultaneously updates its global context and current logical structure based on the generated actions. Experiments on ChCatExt and HierDoc datasets demonstrate the superior performance of Seg2Act in both supervised and transfer learning settings.



## Usage

We provide the code for conducting experiments on the ChCatExt and HierDoc datasets. As an example, we demonstrate how to perform experiments on ChCatExt below. To experiment with Seg2Act on HierDoc, simply download the data from the [MTD](https://github.com/Pengfei-Hu/MTD) repository and follow similar instructions as for ChCatExt.

### Environment Setup

``` bash
conda create -n seg2act python=3.10
conda activate seg2act

pip install -e .
```

### Data Preparation

#### 1. Download and unzip the ChCatExt dataset as below:

``` bash
cd playground/data/
wget https://github.com/Spico197/CatalogExtraction/releases/download/data-v1/ChCatExt.zip
unzip ChCatExt.zip
```

#### 2. Process the train-set and dev-set of ChCatExt into the formats required for Seg2Act-T and Seg2Act, respectively.

``` bash
python seg2act/data/preprocess/ChCatExt/seg2act_t_format.py \
    --input_file playground/data/ChCatExt/[Domain]/train.jsonl \
    --output_file playground/data/ChCatExt/[Domain]/train_seg2act_t_format.json

python seg2act/data/preprocess/ChCatExt/seg2act_format.py \
    --input_file playground/data/ChCatExt/[Domain]/train.jsonl \
    --output_file playground/data/ChCatExt/[Domain]/train_seg2act_3segments_format.json \
    --max_seq 3
```

> Here, [Domain] represents the various document categories of the ChCatExt dataset, which can be BidAnn, CreRat, FinAnn, or DomainMix (representing a combination of all category corpora).
> Additionally, the max_seq parameter represents the input window size for Seg2Act; for further details, please refer to the paper.

#### 3. Process the test-set of ChCatExt into the required format.

``` bash
python seg2act/data/preprocess/ChCatExt/generate_test.py \
    --input_file playground/data/ChCatExt/[Domain]/test.jsonl \
    --output_file playground/data/ChCatExt/[Domain]/test.json
```

### Model Training

Fine-tune the Seg2Act-T and Seg2Act models based on LoRA technology. Below are the instruction for training a Seg2Act-T model, using [Baichuan-7B](https://huggingface.co/baichuan-inc/Baichuan-7B) as the base model on the DomainMix corpora of ChCatExt.

``` bash
python seg2act/train/train_lora.py --seed 17 --base_model baichuan-inc/Baichuan-7B \
 --data_path playground/data/ChCatExt/DomainMix/train_seg2act_t_format.json \
  --output_dir playground/output/ChCatExt/DomainMix/seg2act_t \
```

Below are the instructions for training a Seg2Act model with an input window size of 3, using the same base model and train-set as mentioned above.

``` bash
python seg2act/train/train_lora.py --seed 17 --base_model baichuan-inc/Baichuan-7B \
 --data_path playground/data/ChCatExt/DomainMix/train_seg2act_3segments_format.json \
  --output_dir playground/output/ChCatExt/DomainMix/seg2act_3segments \
```

### Model Evaluation

After training, evaluate the model's performance on the test-set. 
For the Seg2Act-T model, the evaluation instruction on the DomainMix corpora of ChCatExt is shown below.

``` bash
python seg2act/eval/ChCatExt/seg2act_t_eval.py \
    --data_path playground/data/ChCatExt/DomainMix/test.json \
    --base_model baichuan-inc/Baichuan-7B \
    --exp_dir [checkpoint_path] --pred_name pred-DomainMix
```

> Here, [checkpoint_path] refers to the path of the LoRA weights for the fine-tuned model. 

For the Seg2Act model, the evaluation instruction on the DomainMix corpora of ChCatExt is as follows.

``` bash
python seg2act/eval/ChCatExt/seg2act_eval.py \
    --data_path playground/data/ChCatExt/DomainMix/test.json \
    --base_model baichuan-inc/Baichuan-7B \
    --exp_dir [checkpoint_path] --pred_name pred-DomainMix \
    --max_seq 3 --stride 3
```

> The max_seq parameter specifies the input window size designated during the training of Seg2Act, while the stride parameter indicates the output window size during inference. Please note that the output window size must be between 1 and the input window size.

Please note that the current evaluation code is applicable only to models trained with the [Baichuan-7B](https://huggingface.co/baichuan-inc/Baichuan-7B) base model, as we implement a `LogitsProcessor` mechanism based on the Baichuan-7B tokenizer's vocabulary. If support for other models is needed, it can be implemented following similar rules.

## Citation


