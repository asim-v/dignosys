# 1 
FROM python:3.7

# 2
RUN pip install gunicorn certifi==2020.6.20 click==7.1.2 cycler==0.10.0 dataclasses==0.6 decorator==4.4.2 Flask==1.1.2 itsdangerous==1.1.0 Jinja2==2.11.2 joblib==0.17.0 kiwisolver==1.2.0 MarkupSafe==1.1.1 matplotlib==3.3.2 networkx==2.5 numpy==1.19.2 pandas==1.1.3 Pillow==8.0.0 pymongo==3.11.0 pyparsing==2.4.7 python-dateutil==2.8.1 pytz==2020.1 scikit-learn==0.23.2 scipy==1.5.3 six==1.15.0 sklearn==0.0 threadpoolctl==2.1.0 Werkzeug==1.0.1 

# 3
COPY / /app
WORKDIR /app

# 4
ENV PORT 8080

# 5
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 app:app