FROM python:3 

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y 

WORKDIR /microservice

RUN pip install --upgrade pip

COPY ./requirements.txt /microservice/requirements.txt
RUN pip install -r /microservice/requirements.txt

COPY . .

EXPOSE 8000

CMD ["gunicorn", "--bind", "0.0.0.0:8000", "core.wsgi:application"]
