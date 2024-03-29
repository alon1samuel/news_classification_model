FROM python:3
ADD requirements.txt /
RUN pip install -r requirements.txt
COPY src src
ADD notebooks /
CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]