# Use an official Python runtime as a parent image
FROM python:3.10

# Set the working directory in the container to /app
WORKDIR /app

# Update the base image
RUN apt-get update -qq

# Install dependencies
RUN apt-get install -y --fix-missing \
    python3-pip \
    python3-dev \
    build-essential

# Copy app files
COPY . .

# Install dependencies
#RUN pip install --no-cache-dir -r requirements.txt
#RUN pip install -r requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 8501 available to the world outside this container
EXPOSE 8501

# Run app.py when the container launches
CMD streamlit run app.py

