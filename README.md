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
* '''python
* python gCoSRNA_main.py \
  --seq GCGGAUUUAGCUCAGUUGGGAGAGCGCCAGACUGAAGAUCUGGAGGUCCUGUGUUCGAUCCACAGAAUUCGCACCA \
  --sec "(((((((..((((........)))).((((.........)))).....(((((.......))))))))))))...." \
  --model gCoSRNA_model.pkl
'''
* seq: RNA sequence (e.g., "GGACCUCCCGUCCUUGGACGGUCGAGCGAAAGCUUGUGAUUGGUCCG")
* sec: RNA secondary structure in dot-bracket notation (e.g., "(((((..(((((....)))))..((((....))))......)))))")
* model: the trained RF model (i.e., gCoSRNA_model.pkl). Please give the right path for the file.
## 📤 Output
-----------------
A list of predicted coaxially stacked stem pairs, such as:
* Predicted coaxial stacking pairs: H1-H2 (Each H# refers to a stem in the input structure, named from 5′ to 3′ order based on sequence position.)
## Demo

## Training

## 📄 Citation
1. Li S, Xu, Q, Tan YL, Jiang J, YZ Shi, Zhang B. gCoSRNA: Generalizable Coaxial-Stacking Prediction for RNA Junctions Using Secondary Structure. BioRxiv, 2025.

## 📬 Contact
For questions, bug reports, or contributions, please contact:
Ya-Zhou Shi
School of Mathematics & Statistics, Wuhan Textile University
📧 yzshi@wtu.edu.cn
