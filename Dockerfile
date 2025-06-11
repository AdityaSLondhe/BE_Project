# This sets up the container with Python 3.11 installed.
FROM python:3.11.9-slim

# Install necessary system packages for OpenCV and Streamlit
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# This copies everything in your current directory to the /app directory in the container.
COPY . /app

# This sets the /app directory as the working directory for any RUN, CMD, ENTRYPOINT, or COPY instructions that follow.
WORKDIR /app

# This runs pip install for all the packages listed in your requirements.txt file.
RUN pip install -r requirements.txt

# This tells Docker to listen on port 80 at runtime.
EXPOSE 8501

# This command creates a .streamlit directory in the home directory of the container.
RUN mkdir ~/.streamlit

# This copies your Streamlit configuration file into the .streamlit directory you just created.
RUN cp config.toml ~/.streamlit/config.toml
RUN cp secrets.toml ~/.streamlit/secrets.toml


CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]
