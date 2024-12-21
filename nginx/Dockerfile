# Use an official Nginx image as the base image
FROM nginx:latest

# Remove default nginx configuration
RUN rm /etc/nginx/conf.d/default.conf

# Copy custom nginx configuration
COPY nginx.conf /etc/nginx/conf.d/

# Create directory for static files
RUN mkdir -p /app/static

# Set proper permissions
RUN chmod -R 755 /app/static
