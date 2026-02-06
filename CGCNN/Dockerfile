# Use an official lightweight Python image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

COPY requirements.txt .
# Copy requirements first (for caching)
RUN pip install --default-timeout=300 --retries 5 -r requirements.txt


# Copy all project files into the container
COPY . .

# Set the default command to run your program
# (change "main.py" to your actual entry file)
CMD ["python", "-u", "init.py"]