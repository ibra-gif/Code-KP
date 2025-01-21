import tkinter as tk
from tkinter import ttk, messagebox
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import json
import os
import numpy as np

# File untuk menyimpan data
FILE_DATA = "data_penjualan.json"

# Fungsi untuk memuat data dari file
def load_data():
    if os.path.exists(FILE_DATA):
        with open(FILE_DATA, "r") as file:
            return json.load(file)
    return []

# Fungsi untuk menyimpan data ke file
def save_data(data):
    with open(FILE_DATA, "w") as file:
        json.dump(data, file)

# Halaman Profil
def show_profil():
    root.geometry("800x600")
    clear_frame()
    profil_label = tk.Label(frame_main, text="Profil", font=("Helvetica", 20))
    profil_label.pack(pady=20)

    deskripsi_label = tk.Label(frame_main, text=(
        "Aplikasi ini digunakan untuk melakukan prediksi penjualan berbasis data historis. "
        "Menggunakan algoritma Linear Regression untuk memberikan estimasi terhadap penjualan masa depan."),
        font=("Helvetica", 12), wraplength=400, justify="center"
    )
    deskripsi_label.pack(pady=10)

    nama_label = tk.Label(frame_main, text="Langtolang.Cak merupakan sebuah restoran tradisional yang menyediakan beragam menu yang khas dari jawa timur.", font=("Helvetica", 10))
    nama_label.pack(pady=5)

    nim_label = tk.Label(frame_main, text="Lokasi : Jl. Sepakat 2, Bansir Laut, Kec. Pontianak Tenggara, Kota Pontianak", font=("Helvetica", 12))
    nim_label.pack(pady=5)

    back_button = ttk.Button(frame_main, text="Kembali ke Menu Utama", command=show_home)
    back_button.pack(pady=10)

# Halaman Prediksi Penjualan
def show_prediksi_menu():
    root.geometry("800x600")
    clear_frame()

    # Form Input Data
    frame_input = ttk.Frame(frame_main)
    frame_input.pack(pady=10)

    ttk.Label(frame_input, text="Bulan:").grid(row=0, column=0, padx=5)
    entry_bulan = ttk.Entry(frame_input)
    entry_bulan.grid(row=0, column=1, padx=5)

    ttk.Label(frame_input, text="Penjualan:").grid(row=1, column=0, padx=5)
    entry_penjualan = ttk.Entry(frame_input)
    entry_penjualan.grid(row=1, column=1, padx=5)

    def tambah_data_penjualan():
        try:
            bulan = entry_bulan.get().strip()
            penjualan = float(entry_penjualan.get())
            if not bulan:
                raise ValueError("Bulan tidak boleh kosong.")

            data = {"Bulan": bulan, "Penjualan": penjualan}
            data_penjualan.append(data)
            save_data(data_penjualan)
            tampilkan_data_penjualan()
            messagebox.showinfo("Sukses", "Data berhasil ditambahkan!")
        except ValueError as e:
            messagebox.showerror("Error", f"Input tidak valid: {e}")

    ttk.Button(frame_input, text="Tambah Data", command=tambah_data_penjualan).grid(row=2, column=0, columnspan=2, pady=10)

    # Tabel Data Penjualan
    tree = ttk.Treeview(frame_main, columns=("No", "Bulan", "Penjualan"), show="headings")
    tree.pack(pady=10)
    tree.heading("No", text="No")
    tree.heading("Bulan", text="Bulan")
    tree.heading("Penjualan", text="Penjualan")

    def tampilkan_data_penjualan():
        for item in tree.get_children():
            tree.delete(item)
        for idx, record in enumerate(data_penjualan):
            tree.insert("", "end", values=(idx + 1, record["Bulan"], record["Penjualan"]))

    ttk.Button(frame_main, text="Tampilkan Data", command=tampilkan_data_penjualan).pack(pady=5)

    def hapus_data_penjualan():
        try:
            selected_item = tree.selection()
            if not selected_item:
                raise ValueError("Pilih data yang ingin dihapus.")
            
            for item in selected_item:
                values = tree.item(item, "values")
                index = int(values[0]) - 1
                del data_penjualan[index]

            save_data(data_penjualan)
            tampilkan_data_penjualan()
            messagebox.showinfo("Sukses", "Data berhasil dihapus!")
        except ValueError as e:
            messagebox.showerror("Error", f"Hapus data gagal: {e}")

    ttk.Button(frame_main, text="Hapus Data", command=hapus_data_penjualan).pack(pady=5)

    def prediksi_penjualan():
        try:
            if len(data_penjualan) < 2:
                raise ValueError("Data penjualan harus lebih dari 1 untuk prediksi.")
            
            # Data untuk pelatihan model
            X = np.array([int(record["Bulan"]) for record in data_penjualan], dtype=np.int64).reshape(-1, 1)
            y = np.array([record["Penjualan"] for record in data_penjualan], dtype=np.float64)
            
            # Membuat model regresi linear
            model = LinearRegression()
            model.fit(X, y)
    
            # Parameter model
            a = model.intercept_
            b = model.coef_[0]
    
            # Prediksi bulan berikutnya
            bulan_baru = int(X.flatten().max()) + 1
            prediksi = float(model.predict([[bulan_baru]])[0])  # Pastikan float Python
    
            # Evaluasi Model
            y_pred = model.predict(X)  # Prediksi pada data pelatihan
            mae = np.mean(np.abs(y - y_pred))  # Mean Absolute Error
            mse = np.mean((y - y_pred) ** 2)  # Mean Squared Error
            r2 = model.score(X, y)  # R-squared
    
            # Menampilkan hasil
            rincian = (
                f"**Hasil Prediksi Penjualan**\n\n"
                f"1. **Persamaan Regresi:**\n"
                f"   - y = a + b * x\n"
                f"   - a (Intercept): {a:.2f}\n"
                f"   - b (Slope): {b:.2f}\n\n"
                f"2. **Prediksi Bulan Baru (Bulan ke-{bulan_baru}):**\n"
                f"   - Penjualan diprediksi sebesar: {prediksi:.2f}\n\n"
                f"3. **Evaluasi Model:**\n"
                f"   - MAE (Mean Absolute Error): {mae:.2f}\n"
                f"   - MSE (Mean Squared Error): {mse:.2f}\n"
                f"   - R² Score (Koefisien Determinasi): {r2:.2f}\n"
            )
            messagebox.showinfo("Detail Prediksi", rincian)
    
        except ValueError as e:
            messagebox.showerror("Error", f"Prediksi gagal: {e}")


    ttk.Button(frame_main, text="Prediksi Penjualan", command=prediksi_penjualan).pack(pady=5)

    def tampilkan_perhitungan_rinci():
        try:
            if len(data_penjualan) < 2:
                raise ValueError("Data penjualan harus lebih dari 1 untuk perhitungan rinci.")
    
            # Konversi data bulan ke integer
            X = np.array([int(record["Bulan"]) for record in data_penjualan], dtype=np.int64).reshape(-1, 1)
            y = np.array([record["Penjualan"] for record in data_penjualan], dtype=np.float64)
    
            # Membuat model regresi linear
            model = LinearRegression()
            model.fit(X, y)
    
            # Parameter regresi
            a = model.intercept_
            b = model.coef_[0]
    
            # Prediksi untuk bulan berikutnya
            bulan_prediksi = int(X.flatten().max()) + 1  # Mengambil nilai bulan terakhir
            y_prediksi = float(model.predict([[bulan_prediksi]])[0])  # Pastikan tipe float Python
    
            # Konversi data menjadi tipe Python bawaan untuk ditampilkan
            X_list = [int(x) for x in X.flatten()]  # Ubah np.array ke list dengan elemen tipe int
            y_list = [float(val) for val in y]      # Ubah np.array ke list dengan elemen tipe float
    
            rincian = (
                f"**Langkah Perhitungan Prediksi**\n\n"
                f"1. **Data yang Digunakan:**\n"
                f"   - Bulan (X): {X_list}\n"
                f"   - Penjualan (Y): {y_list}\n\n"
                f"2. **Persamaan Regresi:**\n"
                f"   - y = a + b * x\n"
                f"   - a (Intercept): {a:.2f}\n"
                f"   - b (Slope): {b:.2f}\n\n"
                f"3. **Prediksi untuk Bulan ke-{bulan_prediksi}:**\n"
                f"   - x = {bulan_prediksi}\n"
                f"   - y = {a:.2f} + {b:.2f} * {bulan_prediksi}\n"
                f"   - y = {y_prediksi:.2f}\n\n"
                f"**Hasil Prediksi:**\n"
                f"Penjualan pada bulan ke-{bulan_prediksi} diprediksi sebesar {y_prediksi:.2f}"
            )
    
            # Tampilkan hasil perhitungan rinci dalam dialog
            messagebox.showinfo("Perhitungan Rinci Prediksi", rincian)
    
            # Tampilkan grafik
            fig, ax = plt.subplots()
    
            # Data aktual
            ax.scatter(X_list, y_list, color="blue", label="Data Aktual")
            
            # Garis regresi
            x_range = np.arange(X_list[0], bulan_prediksi + 1).reshape(-1, 1)
            ax.plot(x_range, model.predict(x_range), color="red", label="Regresi Linear")
    
            # Prediksi bulan berikutnya
            ax.scatter([bulan_prediksi], [y_prediksi], color="green", label="Prediksi", zorder=5)
    
            ax.set_title("Grafik Prediksi Penjualan")
            ax.set_xlabel("Bulan")
            ax.set_ylabel("Penjualan")
            ax.legend()
    
            # Tampilkan grafik pada GUI
            canvas = FigureCanvasTkAgg(fig, master=frame_main)
            canvas.draw()
            canvas.get_tk_widget().pack(pady=10)
    
        except ValueError as e:
            messagebox.showerror("Error", f"Gagal menampilkan perhitungan rinci: {e}")



    ttk.Button(frame_main, text="Perhitungan Rinci", command=tampilkan_perhitungan_rinci).pack(pady=5)

    back_button = ttk.Button(frame_main, text="Kembali ke Menu Utama", command=show_home)
    back_button.pack(pady=10)

# Halaman Home
def show_home():
    root.geometry("600x400")
    clear_frame()
    home_label = tk.Label(frame_main, text="Selamat Datang di Aplikasi Prediksi Penjualan", font=("Helvetica", 16))
    home_label.pack(pady=20)

    btn_profil = ttk.Button(frame_main, text="Profil", command=show_profil, width=35)
    btn_profil.pack(pady=10)

    btn_prediksi = ttk.Button(frame_main, text="Prediksi Penjualan", command=show_prediksi_menu, width=35)
    btn_prediksi.pack(pady=10)

# Membersihkan Frame
def clear_frame():
    for widget in frame_main.winfo_children():
        widget.destroy()

# Data Penjualan
data_penjualan = load_data()

# GUI
root = tk.Tk()
root.title("Aplikasi Prediksi Penjualan")
root.geometry("800x600")

frame_main = tk.Frame(root)
frame_main.pack(fill=tk.BOTH, expand=True)

show_home()

root.mainloop()
