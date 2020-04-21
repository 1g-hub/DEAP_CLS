FROM pytorch/pytorch:1.1.0-cuda10.0-cudnn7.5-runtime
LABEL maintainer="Akira Terauchi <akira090602@gmail.com>"

SHELL ["/bin/bash", "-c"]

ENV ROOTHOME /root
ENV WORKSPACE /var/www

RUN mkdir -p $WORKSPACE
WORKDIR $WORKSPACE

RUN apt-get update && apt-get install -y libopencv-dev

RUN curl -sSL https://raw.githubusercontent.com/sdispater/poetry/master/get-poetry.py | python

# set poetry's path
ENV PATH $ROOTHOME/.poetry/bin:$PATH

COPY pyproject.toml $WORKSPACE
COPY poetry.lock $WORKSPACE

RUN pip install pyyaml --ignore-installed
RUN pip install --upgrade keyrings.alt
RUN poetry config virtualenvs.create false && \
    pip install --upgrade pip && \
    pip install -U setuptools && \
    poetry install -n

ENV DEBCONF_NOWARNINGS yes
ENV USERNAME python
RUN groupadd -r $USERNAME
RUN useradd -r -g $USERNAME $USERNAME
# RUN chown $USERNAME:$USERNAME -R $ROOTHOME

RUN apt-get update && apt-get install -y locales
RUN locale-gen ja_JP.UTF-8
ENV LANG ja_JP.UTF-8
ENV LANGUAGE ja_JP.UTF-8  
ENV LC_ALL ja_JP.UTF-8

# USER $USERNAME
WORKDIR workspace

CMD ["python"]
