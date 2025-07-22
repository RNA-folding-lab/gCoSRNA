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
1. Command-line Script
* The main script for prediction is gCoSRNA-predict.py. It takes three arguments:
* '''python
* python gCoSRNA_main.py \
  --seq GCGGAUUUAGCUCAGUUGGGAGAGCGCCAGACUGAAGAUCUGGAGGUCCUGUGUUCGAUCCACAGAAUUCGCACCA \
  --sec "(((((((..((((........)))).((((.........)))).....(((((.......))))))))))))...." \
  --model gCoSRNA_model.pkl
'''
* --seq: RNA sequence (e.g., "GGACCUCCCGUCCUUGGACGGUCGAGCGAAAGCUUGUGAUUGGUCCG")
* --sec: RNA secondary structure in dot-bracket notation (e.g., "(((((..(((((....)))))..((((....))))......)))))")
* --model: the trained RF model (i.e., gCoSRNA_model.pkl). Please give the right path for the file.
2. Jupyter Notebook
* You can also use the notebook version: gCoSRNA_predict.ipynb.
* Simply open the notebook, modify the input RNA sequence and secondary structure in the designated cell, and run all cells to get the prediction results.
## 📤 Output
-----------------
A list of predicted coaxially stacked stem pairs, such as:
* Predicted coaxial stacking pairs: H1-H2 (Each H# refers to a stem in the input structure, named from 5′ to 3′ order based on sequence position.)
## 🧪 Demo
This package includes demo examples covering 2-way, 3-way, and 4-way RNA junctions. Each example includes:
* The RNA sequence
* The corresponding dot-bracket secondary structure
* Output from the program, including identified loops, stems, extracted substructures, and predicted coaxial stacking
## 🔧 Model Training (Random Forest)
If you would like to retrain the model, a training script is provided.
* Open the gCoSRNA-train.py file.
* Locate the line marked with # ----- you need to change -----.
* Modify the file path to point to your own TrainingData.xlsx file.
* Run the script: python Train.py
* After training, the model will be saved as: gCoSRNA_model.pkl

## 📄 Citation
1. Li S, Xu, Q, Tan YL, Jiang J, YZ Shi, Zhang B. gCoSRNA: Generalizable Coaxial-Stacking Prediction for RNA Junctions Using Secondary Structure. BioRxiv, 2025.
2. Laing C, Wen D, Wang JT, Schlick T. Predicting coaxial helical stacking in RNA junctions. Nucleic Acids Res, 2012;40(2):487-498.

## 📬 Contact
For questions, bug reports, or contributions, please contact:
Ya-Zhou Shi
School of Mathematics & Statistics, Wuhan Textile University
📧 yzshi@wtu.edu.cn
