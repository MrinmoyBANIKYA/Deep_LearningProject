import subprocess
import os
import sys

def run_script(script_path):
    print(f"\n" + "="*60)
    print(f"RUNNING: {script_path}")
    print("="*60)
    
    # Get the directory and filename
    dirname = os.path.dirname(script_path)
    filename = os.path.basename(script_path)
    
    # Run the script with the research directory as CWD if it's in research/
    cwd = dirname if dirname else "."
    
    result = subprocess.run([sys.executable, filename], cwd=cwd)
    
    if result.returncode != 0:
        print(f"\nERROR: {script_path} failed with exit code {result.returncode}")
        return False
    return True

def main():
    scripts = [
        "research/generate_dataset.py",
        "research/logistic_baseline.py",
        "research/xgboost_baseline.py",
        "research/tabnet_model.py",
        "research/hybrid_framework.py",
        "research/ablation_study.py",
        "research/evaluation_utils.py"
    ]
    
    for script in scripts:
        if not os.path.exists(script):
            print(f"Warning: Script {script} not found. Skipping...")
            continue
            
        success = run_script(script)
        if not success:
            print("\nPipeline halted due to error.")
            sys.exit(1)
            
    print("\n" + "="*60)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("="*60)

if __name__ == "__main__":
    main()
