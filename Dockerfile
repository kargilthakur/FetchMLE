# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory to /app
WORKDIR /app

COPY requirements.txt ./requirements.txt

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

EXPOSE 8502

HEALTHCHECK CMD curl --fail http://localhost:8502/_stcore/health

# Start the Streamlit app
CMD ["/bin/bash", "-c", "python -m unittest discover tests && python main.py && streamlit run streamlit/app.py --server.port=8502 --server.address=0.0.0.0"]
