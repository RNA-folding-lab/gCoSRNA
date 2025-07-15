# gCoSRNA
Generalizable Coaxial-Stacking Prediction for RNA Junctions Using Secondary Structure.
## Introduction
----------------
gCoSRNA is a generalizable and topology-independent framework for predicting coaxial helical stacking in RNA multi-way junctions using only secondary structure-derived features. By decomposing complex junctions into pseudo two-way stem pairs and applying a unified machine learning model, gCoSRNA captures shared stacking signatures across diverse junction types and achieves accurate predictions without relying on junction-specific classifications.
## 🔧 Dependencies
-------------------
Before running `gCoSRNA`, make sure the following Python packages are installed:
* pandas, joblib, math, draw_rna, os [You can install the required dependencies via: pip install pandas joblib]
* The draw_struct module is optional and used only for visualization.
## 🖥 Environment
-----------------
* Tested under Linux (Ubuntu 20.04) with Python 3.7+
* Notebook version is available and tested on Windows
* Recommended: Use in a conda environment or virtualenv for clean dependency management
## 🚀 Usage
-----------------
* The main script for prediction is gCoSRNA-predict.py. It takes two arguments:
'''bash
python gCoSRNA-predict.py $seq $sec
'''
* seq: RNA sequence (e.g., "GGACCUCCCGUCCUUGGACGGUCGAGCGAAAGCUUGUGAUUGGUCCG")
* sec: RNA secondary structure in dot-bracket notation (e.g., "(((((..(((((....)))))..((((....))))......)))))")
## 📤 Output
-----------------
A list of predicted coaxially stacked stem pairs, such as:
* Predicted coaxial stacking pairs: H1-H2 (Each H# refers to a stem in the input structure, named from 5′ to 3′ order based on sequence position.)
