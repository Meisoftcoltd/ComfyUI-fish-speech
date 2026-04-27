import sys
import os
import subprocess

def install_requirements():
    req_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "requirements.txt")
    if os.path.exists(req_file):
        print("Instalando requerimientos para Fish Speech...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', req_file])

if __name__ == "__main__":
    install_requirements()
