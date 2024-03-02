FROM python:3.9.0
COPY . .

RUN pip install -r requirements.txt

ENTRYPOINT [ "python" ]

EXPOSE 5000

CMD [ "app.py" ]