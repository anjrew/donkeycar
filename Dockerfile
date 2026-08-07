FROM python:3.9

WORKDIR /app

# setup.cfg resolves its version dynamically from the donkeycar package
# (version = attr: donkeycar.__version__), so pip needs the full source
# tree present before it can even read the build metadata -- it can't be
# installed from setup.py/README alone ahead of the rest of the source.
ADD . /app

# install donkey with tensorflow (cpu only version)
RUN pip install -e .[pc]

# get testing requirements
RUN pip install -e .[dev]

# setup jupyter notebook to run without password
RUN pip install jupyter notebook
RUN jupyter notebook --generate-config
RUN echo "c.NotebookApp.password = ''">>/root/.jupyter/jupyter_notebook_config.py
RUN echo "c.NotebookApp.token = ''">>/root/.jupyter/jupyter_notebook_config.py

#start the jupyter notebook
CMD jupyter notebook --no-browser --ip 0.0.0.0 --port 8888 --allow-root  --notebook-dir=/app/notebooks

#port for donkeycar
EXPOSE 8887

#port for jupyter notebook
EXPOSE 8888
