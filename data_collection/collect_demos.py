"""
Demonstration Data Collection Pipeline.
"""

import argparse

def main():
    parser = argparse.ArgumentParser(description="Collect demonstration episodes into LeRobot format.")
    parser.add_argument("--num-episodes", type=int, default=50, help="Total episodes to collect")
    parser.add_argument("--output-dir", type=str, default="./data", help="Output dataset directory")
    args = parser.parse_args()
    print(f"Collecting {args.num_episodes} demonstration episodes...")

if __name__ == "__main__":
    main()
