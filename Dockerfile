FROM tensorflow/syntaxnet
COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt
COPY Russian-SynTagRus /models/Russian-SynTagRus
ENV PYTHONPATH $PYTHONPATH:/opt/tensorflow/syntaxnet/bazel-bin/dragnn/tools/oss_notebook_launcher.runfiles:/opt/tensorflow/syntaxnet/bazel-bin/dragnn/tools/oss_notebook_launcher.runfiles/protobuf/python:/opt/tensorflow/syntaxnet/bazel-bin/dragnn/tools/oss_notebook_launcher.runfiles/__main__:/opt/tensorflow/syntaxnet/bazel-bin/dragnn/tools/oss_notebook_launcher.runfiles/six_archive:/opt/tensorflow/syntaxnet/bazel-bin/dragnn/tools/oss_notebook_launcher.runfiles/org_tensorflow:/opt/tensorflow/syntaxnet/bazel-bin/dragnn/tools/oss_notebook_launcher.runfiles/protobuf
EXPOSE 9000
COPY main.py /app/main.py
CMD ["/app/main.py"]
