In order to run the notebook contained in this folder
-create a virtual environment:
(I used virtualenvs with python 3.6.9) 
python3.6 -m venv <name_of_environment>

-enter the environment and install dependencies:
pip install -r requirements.txt

-install the kernel in order that it is available to ipython
ipython kernel install --user --name=<name_of_environment>

Now when you launch the notebook using jupyter the kernel should be available to you.


