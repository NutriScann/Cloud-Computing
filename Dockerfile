# Gunakan base image Python
FROM python:3.11-slim

# Atur direktori kerja di dalam container
WORKDIR /app

# Salin file requirements.txt ke direktori kerja
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Salin semua file dari proyek Anda ke direktori kerja
COPY . .

# Expose port yang akan digunakan oleh Flask
EXPOSE 5000

# Tentukan perintah untuk menjalankan aplikasi Flask
CMD ["python", "run.py"]
