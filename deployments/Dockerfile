FROM python:3.6

ADD app/ /app
ADD deployments/ /app
WORKDIR /app/
RUN ls -lrt
#COPY /deploymnts/requirements.txt /app/
RUN pip install -r ./requirements.txt

COPY app/house_price.py app/pricing_api.py app/housing_model.pkl app/housing_model_columns.pkl /app/

# ENTRYPOINT /bin/bash
EXPOSE 7676

#ENV ENVIRONMENT local

ENTRYPOINT python ./pricing_api.py
#python ./pricing_api.py
