FROM python:3.12.3-slim
WORKDIR /app
COPY requirement.txt .
RUN pip install --no-cache-dir -r requirement.txt
COPY translator_api.py .
EXPOSE 5000
CMD ["python","translator_api.py"]
