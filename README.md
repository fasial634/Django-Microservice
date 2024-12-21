# Microservice for AI Solution
## Overview
This microservice is designed to provide AI-driven solutions for AI and NLP solution. It is built using Django framework. 

```bash
git clone https://github.com/fasial634/Django-Microservice.git
```
```bash
sudo docker network create django_network 
```
## PostgreSQL
```bash
docker run -d --name postgres-db --network django_network -e POSTGRES_DB=mainDatabase -e POSTGRES_USER=root -e POSTGRES_PASSWORD='12345' -p 5432:5432 postgres
```

## Django 
```bash
cd microservice/
sudo docker build -t django-microservice .
sudo docker run -d  --name django_container --env-file .env --network django_network  -v django-static:/micro_service/staticfiles django-microservice
```
```bash
# Check container logs
docker logs django_container

# Access Django container shell
docker exec -it django_container bash

cd microservice/

# Run migrations
python manage.py makemigrations
python manage.py migrate
```
## Nginx
```bash
cd nginx/
sudo docker build -t nginx-proxy . 
sudo docker run -d  --name nginx_container  --network django_network  -v django-static:/app/static  -p 80:80  nginx-proxy
```


