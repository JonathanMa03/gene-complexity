"""Run the separate nested donor-held-out robustness experiment."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code/robustness'))
from donor_study import main
if __name__=='__main__': main()
