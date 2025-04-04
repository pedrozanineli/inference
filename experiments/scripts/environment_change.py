import os
import sys
import subprocess

calculator,model = sys.argv[1],sys.argv[2]

def activate_environment(calculator):

    activate_cmd = (
            f"source ~/.bashrc &&"
            f"source $(conda info --base)/etc/profile.d/conda.sh && "
            f"conda activate {calculator} && "
            # f"conda info --env"
            f"python3 run_inference.py {calculator} {model}"
    )
    subprocess.run(activate_cmd, shell=True, executable="/bin/bash")

activate_environment(calculator)

