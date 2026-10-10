"""G3.1 CLI skeleton — collect / evaluate / report.

Confirm evaluation blocked until threshold lock.
"""
import argparse


def main():
    p = argparse.ArgumentParser()
    p.add_argument('cmd', choices=['collect', 'evaluate', 'report', 'lock-threshold'])
    args = p.parse_args()
    if args.cmd == 'evaluate':
        print("G3.1 evaluate: infrastructure ready; confirm blocked until pilot + threshold lock.")
    elif args.cmd == 'lock-threshold':
        print("Threshold remains TO_LOCK — not set by this run.")
    else:
        print(f"G3.1 {args.cmd}: skeleton ready.")


if __name__ == '__main__':
    main()
