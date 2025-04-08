# DPPBP: Dual-stream Protein-peptide Binding Sites Prediction Based on Region Detection
A novel Dual-stream Protein-Peptide Binding sites Prediction method (DPPBP) based on region detection and protein language model. To use it, firstly, you need to download protBERT file from the following URL https://huggingface.co/Rostlab/prot_bert_bfd.

## Requirements
python==3.8.10
pytorch==2.2.2
scipy==1.10.1
transformers==4.46.3

## Dataset
The dataset used in this project comes from PepBCL.

## Usage
### Training the stream 1 module:
```python main.py```

### Prediction the stream 1 module:
tester = Tester(model, data, args)  
```python main.py```  
Then, the stream 1 module will generate a .txt file containing the predicted binding site information.

### Training the stream 2 module:
```python /train/protBert_main.py```

### Evaluation
The model computes the union of predictions from both streams as the final binding site prediction for evaluation.
