# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Run unit tests, then run the main script, and finally run the Streamlit app
CMD ["sh", "-c", "python -m unittest discover tests && python main.py && streamlit run streamlit/app.py"]
