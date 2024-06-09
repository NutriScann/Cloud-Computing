
---

# Aplikasi Prediksi Makanan dengan Flask

Aplikasi ini menggunakan Flask dan TensorFlow untuk memprediksi jenis makanan dari gambar yang diunggah. Aplikasi ini juga menyediakan daftar makanan yang dapat dikenali beserta deskripsi dan informasi nutrisinya.

## Fitur

- **/predict**: Memprediksi jenis makanan dari gambar yang diunggah dalam format base64.
- **/list**: Menampilkan daftar makanan yang dapat dikenali oleh model.

## Prasyarat

Pastikan Anda memiliki:
- Python 3.11 atau lebih baru
- pip (Python package installer)

## Instalasi

Ikuti langkah-langkah di bawah ini untuk menginstal aplikasi:

1. **Clone repositori ini:**
    ```bash
    git clone https://github.com/NutriScann/cloud-computing.git
    cd repo-name
    ```

2. **Buat virtual environment (opsional tetapi disarankan):**
    ```bash
    python -m venv venv
    source venv/bin/activate # untuk pengguna Linux atau MacOS
    .\venv\Scripts\activate # untuk pengguna Windows
    ```
 
3. **Instal dependensi:**
    ```bash
    pip install -r requirements.txt
    ```

4. **Buat file `.env` dan tambahkan SECRET_KEY:**
    ```env
    SECRET_KEY=your_secret_key
    ```

5. **Pastikan file model `model.h5` berada dalam direktori `app/`.**

## Penggunaan

1. **Jalankan aplikasi Flask:**
    ```bash
    python run.py
    ```
    ```bash
    flask run
    ```

2. **Akses aplikasi melalui `http://localhost:5000`.**

### Endpoint `/predict`

- **Deskripsi**: Memprediksi jenis makanan dari gambar yang diunggah.
- **Metode**: POST
- **Input**: JSON dengan gambar dalam format base64
    ```json
    {
        "image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEAAAAAAAD..."
    }
    ```
- **Output**: JSON dengan status, message, dan data prediksi
    ```json
    {
        "status": 200,
        "message": "success",
        "data": [
            {
                "name": "nasi_goreng",
                "desc": "Deskripsi Nasi Goreng",
                "carbo": "50g",
                "protein": "20g",
                "calory": "500 kkal",
                "lemak": "20g",
                "bahan": ["nasi", "kecap", "telur", "ayam"]
            }
        ]
    }
    ```

### Endpoint `/list`

- **Deskripsi**: Menampilkan daftar makanan yang dapat dikenali oleh model.
- **Metode**: GET
- **Output**: JSON dengan status, message, dan data makanan
    ```json
    {
        "status": 200,
        "message": "success",
        "data": [
            {
                "name": "bakso",
                "desc": "Deskripsi Bakso",
                "carbo": "30g",
                "protein": "10g",
                "calory": "250 kkal",
                "lemak": "15g",
                "bahan": ["daging sapi", "tepung", "bawang putih"]
            },
            {
                "name": "rawon",
                "desc": "Deskripsi Rawon",
                "carbo": "20g",
                "protein": "30g",
                "calory": "400 kkal",
                "lemak": "25g",
                "bahan": ["bebek", "bumbu kuning", "daun pisang"]
            },
        ]
    }
    ```

## Struktur Direktori

```
.
├── app
│   ├── __init__.py
│   ├── routes.py
│   └── model.h5
├── .env
├── requirements.txt
├── run.py
└── config.py
```

## Lisensi

Aplikasi ini dilisensikan di bawah [MIT License](LICENSE).

---