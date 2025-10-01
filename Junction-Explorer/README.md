# Junction-Explorer (Re-implementation)

**Junction-Explorer** is a Random Forest–based method originally developed to predict coaxial stacking configurations in RNA junctions using features derived from secondary structure (Laing et al., 2012).

To facilitate direct comparison with our method gCoSRNA, we re-implemented the RNA Junction-Explorer pipeline following the algorithmic details from the original publication.

* We first replicated the original training using the reported feature sets, training data (e.g., 110 three-way junctions), and hyperparameters.
* We then retrained the model on a newly curated, non-redundant RNA junction dataset, while keeping the original feature definitions unchanged. This retrained version is referred to as **Junction-Explorer-n**.

## Repository Contents

* **Junction-Explorer.py** – Core implementation of the algorithm
* **Junction-Explorer-main.py** – Main script for running predictions
* **Schlick_three_way_model.pkl** – Pretrained model for three-way junctions (original dataset)
* **Schlick_four_way_model.pkl** – Pretrained model for four-way junctions (original dataset)
* **junction-explorer-n-three_way.pkl** – Retrained model for three-way junctions (new dataset)
* **junction-explorer-n-four_way.pkl** – Retrained model for four-way junctions (new dataset)
* **example_seq.txt** – Example RNA sequence input
* **example_sec.txt** – Example RNA secondary structure input (dot-bracket notation)

## Usage

We provide two ways to run the program. All examples assume you are in the repository directory:

### (1) Run with sequence and secondary structure directly in the command line

```bash
cd Junction-Explorer   # Replace with the path to your directory
python Junction-Explorer-main.py \
  --seq GCGGAUUUAGCUCAGUUGGGAGAGCGCCAGACUGAAGAUCUGGAGGUCCUGUGUUCGAUCCACAGAAUUCGCACCA \
  --sec "(((((((..((((........)))).((((.........)))).....(((((.......))))))))))))...."
```

### (2) Run with input files

```bash
cd Junction-Explorer   # Replace with the path to your directory
python Junction-Explorer-main.py \
  --seq_file example_seq.txt \
  --sec_file example_sec.txt
```

The outputs of both execution modes will be displayed in the terminal.

## Using Junction-Explorer-n

To use the retrained **Junction-Explorer-n** models, modify the following lines in **Junction-Explorer.py**:

* Line 689 → replace with the path to `junction-explorer-n-three_way.pkl`
* Line 691 → replace with the path to `junction-explorer-n-four_way.pkl`


## Reference
1. Laing C, Wen D, Wang JT, Schlick T. Predicting coaxial helical stacking in RNA junctions. Nucleic Acids Res, 2012;40(2):487-498.
2. Laing C, Jung S, Kim N, Elmetwaly S, Zahran M, Schlick T. Predicting Helical Topologies in RNA Junctions as Tree Graphs. PLoS ONE. 2013; 8(8): e71947.

