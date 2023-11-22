# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory to /app
WORKDIR /app

# Clone the repository
RUN apt-get update && apt-get install -y git
RUN git clone https://github.com/kargilthakur/FetchMLE .

# Create and activate a virtual environment
RUN python -m venv venv
RUN /bin/bash -c "source venv/bin/activate"

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Run unit tests, then run the main script, and finally run the Streamlit app
CMD ["/bin/bash", "-c", "python main.py && python -m unittest discover tests && streamlit run streamlit/app.py"]
