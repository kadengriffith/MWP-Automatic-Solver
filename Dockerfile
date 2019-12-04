FROM tensorflow/tensorflow:nightly-gpu-py3

WORKDIR /home/translator

RUN pip install --upgrade pip
RUN pip install --upgrade virtualenv
RUN virtualenv /home/translator
RUN . /home/translator/bin/activate
RUN pip install tensorflow-datasets nltk word2number apache-beam mwparserfromhell