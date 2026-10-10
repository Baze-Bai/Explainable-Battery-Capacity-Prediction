# Explainable Battery Capacity Prediction

Research source for battery-capacity regression from multi-temperature electrochemical impedance spectroscopy (EIS). A multi-branch PyTorch 1D CNN processes impedance inputs; Captum Integrated Gradients inspects input contributions. Evaluation helpers calculate MSE and R-squared.

## Implementation

- `Scripts/CNN_Class.py`: `MultiSource1DCNN` with parallel convolutional branches and a regression output.
- `Scripts/Data_Preprocess.py`: measurement loading, capacity normalization, input reshaping and train/validation/test selection.
- `Scripts/Dataset_Class.py`: dataset and loader utilities.
- `Scripts/Multi_Func.py`: training, evaluation and attribution helpers.
- `Show.ipynb`: original training/attribution workflow with all execution results, attachments and user metadata removed. An obsolete import was removed and the checkpoint-load path was aligned with the preceding save cell.

## Setup and data

```bash
python -m pip install -r requirements.txt
```

Open `Show.ipynb` in a Jupyter environment after preparing your own authorized data. `Data_Preprocess` accepts EIS and capacity directory arguments. Its current file lists expect measurements for the battery identifiers and 25/35/45-degree temperature conditions specified in the source. Inspect these lists, column assumptions, 60-frequency reshaping and normalization before using a different dataset. This is not a generic data-schema adapter.

The notebook illustrates training, checkpoint loading and Integrated Gradients. Real measurements, saved models and paper/manuscript files are intentionally absent. The 1D CNN predicts normalized capacity; this release does not claim a validated remaining-useful-life estimator. Attribution maps describe model sensitivity, not physical causality. Only source/notebook syntax and output removal were verified for publication. Dependencies, training, GPU execution, prediction accuracy and attribution results were not rerun.

## Publication scope

This independent sanitized public repository preserves the original private research repository unchanged. It exports source, an output-free notebook and documentation, without original Git history, raw EIS/capacity records, model weights, manuscripts or private result plots. See `NOTICE.md`. The code is MIT-licensed; third-party dependencies and any user-supplied data/models retain their own terms.
