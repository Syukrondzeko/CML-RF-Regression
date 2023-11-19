1. Create virtual environtment
2. Remove all packages to make it clean "pip freeze | xargs pip uninstall -y"
3. Install all packages
4. Try to run train.py
5. Delete image and txt files that appears as outputs
6. Create requirements.txt with "pip freeze > requirements.txt" or "pip freeze | cut -d= -f1 > requirements.txt" if you don't want the versions