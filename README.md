# The Curse of Performance Instability in Analysis Datasets: Consequences, Source, and Suggestions

This is the official repo for the following paper

* The Curse of Performance Instability in Analysis Datasets: Consequences, Source, and Suggestions, Xiang Zhou, Yixin Nie, Hao Tan and Mohit Bansal, EMNLP 2020 ([arxiv](https://arxiv.org/abs/2004.13606))

# Dependencies

This code requires Python 3. All the dependencies are specified in "requirement.txt"

```
pip install -r requirements.txt
```

# Instructions

The current code supports the calculation of decomposed variance metrics from standard evaluation numbers.

1. Download the [NLI datasets](https://drive.google.com/file/d/1PbODxp4uEsZR_pYOKJSu96xB8rwsdklv/view?usp=sharing) and put it under the  `nli_data` folder in the root directory

2. Organize the evaluation result of your model under the `models` directly in the same way as the `berts` (an example folder showing the result of BERT-base) folder, name of the folder representing the model type

   * `MODEL_TYPE/seed_x` saves the evaluation results with seed `x` 
   * Inside `MODEL_TYPE/seed_x/`, each folder represent the evaluation result on one dataset, including three files:
     * `eval_results.txt` : Final accuracy of the model
     * `logits_results.txt` : List of logits output by the model on every example in the dataset
     * `pred_results.txt` : List of labels predicted by the model on every example in the dataset

3. Run the evaluation scripts by 

   ```
   python variance_report.py MODEL_PATH
   ```

   

Other scripts (training/evaluation/analysis) and model checkpoints that are used in the paper will come soon.
