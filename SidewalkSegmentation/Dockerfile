# Use the official Python image as base
FROM python:3.10

# Set the working directory
WORKDIR /code

# Set environment variables for huggingface transformers cache and Matplotlib cache
ENV TRANSFORMERS_CACHE /tmp/huggingface_cache
ENV MPLCONFIGDIR /tmp/matplotlib_cache

# Install required packages directly from GitHub repositories
RUN pip install --no-cache-dir git+https://github.com/huggingface/transformers.git \
                               torchvision \
                               torch \
                               matplotlib \
                               shiny

# Create the directory and set permissions
RUN mkdir ./UploadedImg && chmod 777 ./UploadedImg
RUN mkdir ./Result && chmod 777 ./Result

# Copy the rest of the application code into the container
COPY . .

# Expose port 8000
EXPOSE 7860

# Specify the command to run your application
CMD ["shiny", "run", "app.py", "--host", "0.0.0.0", "--port", "7860"]