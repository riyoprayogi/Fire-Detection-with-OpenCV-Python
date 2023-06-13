import cv2
import numpy as np
import sys
import datetime

from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton, QMessageBox
from PyQt5.QtGui import QPixmap, QImage
from plyer import notification


class FireDetection(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Fire Detection')

        # Mengatur warna latar belakang
        self.setStyleSheet("background-color: #161B22;")
        # 161B22
        # Membuat widget utama
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        # Membuat tata letak utama
        self.layout = QVBoxLayout(self.central_widget)

        # Membuat label untuk memasukkan alamat RTSP
        self.label = QLabel('Alamat RTSP:', self)
        self.label.setStyleSheet("color: white;")
        self.layout.addWidget(self.label)

        # Membuat input teks untuk alamat RTSP
        self.rtsp_input = QLineEdit(self)
        self.rtsp_input.setStyleSheet("background-color: #1E242B; color: white;")
        self.layout.addWidget(self.rtsp_input)

        # Membuat tombol untuk memulai deteksi
        self.start_button = QPushButton('Mulai', self)
        self.start_button.setStyleSheet("background-color: #1E242B; color: white;")
        self.start_button.clicked.connect(self.start_detection)
        self.layout.addWidget(self.start_button)

        # Membuat tombol untuk menghentikan deteksi
        self.stop_button = QPushButton('Stop', self)
        self.stop_button.setEnabled(False)
        self.stop_button.setStyleSheet("background-color: #1E242B; color: white;")
        self.stop_button.clicked.connect(self.stop_detection)
        self.layout.addWidget(self.stop_button)

        # Membuat label untuk menampilkan video
        self.video_label = QLabel(self)
        self.layout.addWidget(self.video_label)

        # Objek VideoCapture untuk mengakses video dari CCTV
        self.cap = None

        # Inisialisasi variabel untuk rekaman citra
        self.recording = False
        self.out = None

    def start_detection(self):
        # Mendapatkan alamat RTSP dari input teks
        rtsp_url = self.rtsp_input.text()

        # Membuka video menggunakan alamat RTSP
        self.cap = cv2.VideoCapture(rtsp_url)

        # Periksa apakah video berhasil diakses
        if not self.cap.isOpened():
            msg_box = QMessageBox()
            msg_box.setStyleSheet("background-color: white; color: #1E242B;")
            msg_box.warning(self, 'Kesalahan', 'Tidak dapat mengakses CCTV', QMessageBox.Ok, QMessageBox.Ok)
            return

        # Mengubah status tombol
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)

        # Define the lower and upper HSV color range for the color of the fire
        lower_color = np.array([-10, -50, 205])
        upper_color = np.array([10, 50, 305])

        # lower_color = np.array([-3, 86, 205])
        # upper_color = np.array([17, 186, 305])

        # Loop untuk membaca frame dari video CCTV
        while True:
            ret, frame = self.cap.read()

            # Periksa apakah frame berhasil dibaca
            if not ret:
                msg_box = QMessageBox()
                msg_box.setStyleSheet("background-color: white; color: #1E242B;")
                msg_box.warning(self, 'Notif', 'Tidak dapat membaca frame atau deteksi telah berhenti', QMessageBox.Ok, QMessageBox.Ok)
                break

            # Convert the frame to the HSV color space
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Create a binary mask of the fire region based on the HSV color range
            mask = cv2.inRange(hsv_frame, lower_color, upper_color)

            # Apply morphological operations to remove noise and smooth the binary mask
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.erode(mask, kernel, iterations=1)
            mask = cv2.dilate(mask, kernel, iterations=2)

            # Find the contours of the fire region in the binary mask
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Draw the contours on the original frame
            for cnt in contours:
                cv2.drawContours(frame, [cnt], 0, (0, 0, 255), 2)

                # Start recording if fire is detected
                if not self.recording:
                    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    file_name = f"C:/Users/MSI KATANA/Videos/IPPL/fire_{current_time}.mp4"  # Ganti dengan alamat penyimpanan yang diinginkan
                    frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    self.out = cv2.VideoWriter(file_name, cv2.VideoWriter_fourcc(*"mp4v"), 25, (frame_width, frame_height))
                    self.recording = True
                    notification.notify(
                        title='Terdeteksi Api',
                        message='Api terdeteksi! Merekam citra...',
                        app_icon=None,
                        timeout=10
                    )


            # Stop recording if fire disappears
            if self.recording and len(contours) == 0:
                self.recording = False
                self.out.release()
                notification.notify(
                    title='Api Hilang',
                    message='Api sudah tidak terdeteksi. Rekaman citra dihentikan.',
                    app_icon=None,
                    timeout=10
                )

            # Menampilkan frame pada label video
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_qimage = QImage(frame_rgb.data, frame_rgb.shape[1], frame_rgb.shape[0], QImage.Format_RGB888)
            frame_pixmap = QPixmap.fromImage(frame_qimage)
            self.video_label.setPixmap(frame_pixmap)

            # Mengupdate tampilan
            QApplication.processEvents()

            # Menulis frame ke file rekaman jika sedang merekam
            if self.recording:
                self.out.write(frame)

            # Tekan tombol q untuk keluar dari loop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Memanggil method stop_detection() ketika deteksi berakhir
        self.stop_detection()

    def stop_detection(self):
        # Mengubah status tombol
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

        # Memastikan objek VideoCapture telah dibuka sebelumnya
        if self.cap is not None and self.cap.isOpened():
            # Membebaskan objek VideoCapture
            self.cap.release()

        # Membebaskan objek VideoWriter jika sedang merekam
        if self.recording:
            self.recording = False
            self.out.release()
            notification.notify(
                title='Rekaman Selesai',
                message='Rekaman citra selesai dan disimpan.',
                app_icon=None,
                timeout=10
            )

    def closeEvent(self, event):
        # Memberikan konfirmasi kepada pengguna saat menutup aplikasi
        reply = QMessageBox.question(self, 'Keluar', 'Apakah Anda ingin keluar dari aplikasi?',
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = FireDetection()
    window.show()
    sys.exit(app.exec_())
