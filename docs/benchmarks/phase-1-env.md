# Faz 1 — Ortam

Bu yazılım ekosistemi Bursa Teknik Üniversitesi Bilgisayar Mühendisliği 1. sınıf öğrencisi Muhammed Fatih Şahin tarafından yapay zeka destekli olarak geliştirilmiştir.

ESP-IDF Python kullanılmadı. `py -3.12` bu makinede yok. Sanal ortam Anaconda Python 3.11 ile açıldı.
Anaconda `python.exe` yanındaki VCRUNTIME140, resmi CPU tekerlerinin DLL başlatmasını düşürdü. `.bench-venv/host` aynı yorumlayıcıyı sistem CRT ile çalıştırır ve `pyvenv.cfg` bu dizini gösterir.

- Python: `3.11.5 | packaged by Anaconda, Inc. | (main, Sep 11 2023, 13:26:23) [MSC v.1916 64 bit (AMD64)]`
- Yorumlayıcı: `C:\Users\muham\Desktop\Nexus\.bench-venv\Scripts\python.exe`
- Platform: `Windows-10-10.0.26200-SP0`
- İşlemci: `AMD Ryzen 7 5800H with Radeon Graphics`
- numpy: `2.4.6`
- torch: `2.14.1+cpu`
- tensorflow: `2.21.0`
- torch.cuda.is_available(): `False`
- tf.config.list_physical_devices('GPU'): `[]`


Sürümler bu sanal ortamdan okundu.
