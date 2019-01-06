FROM jupyter/base-notebook
LABEL maintainer='shihxuancheng@gmail.com'

# copy all contents into jupyter working directory
COPY --chown=1000:100 . /home/jovyan/work/ai_exercise/

# install required packages
RUN pip install -r /home/jovyan/work/ai_exercise/require_pkg.txt

# start jupyter without token
CMD ["start.sh","jupyter","lab","--LabApp.token=''"]
