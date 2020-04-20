FROM tensorflow/tensorflow:nightly-gpu-py3

WORKDIR /home/translator

RUN pip install --upgrade pip
RUN pip install --upgrade virtualenv
RUN virtualenv /home/translator
RUN . /home/translator/bin/activate
RUN pip install tensorflow-datasets
RUN pip install pyyaml
RUN pip install nltk
RUN pip install word2number
RUN pip install pycodestyle
RUN pip install apache-beam
RUN pip install lxml
RUN pip install mwparserfromhell
