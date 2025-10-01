
import argparse
from Junction_Explorer import predict_coaxial_stacking

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="gCoSRNA: Predict coaxial stacking in RNA secondary structures.")
    parser.add_argument("--seq", help="RNA sequence")
    parser.add_argument("--seq_file", help="Path to RNA sequence file")
    parser.add_argument("--sec", help="Dot-bracket secondary structure")
    parser.add_argument("--sec_file", help="Path to dot-bracket file")
    parser.add_argument("--no-vis", action="store_true", help="Disable structure visualization")

    args = parser.parse_args()
    
    
    if args.seq_file:
        with open(args.seq_file, 'r') as f:
            seq = f.read().strip()
    elif args.seq:
        seq = args.seq
    else:
        raise ValueError("Please provide either --seq or --seq_file")

    if args.sec_file:
        with open(args.sec_file, 'r') as f:
            sec = f.read().strip()
    elif args.sec:
        sec = args.sec
    else:
        raise ValueError("Please provide either --sec or --sec_file")

    # prediction
    predict_coaxial_stacking(
        sequence=seq,
        dot_bracket=sec,
        visualize=not args.no_vis
    )
