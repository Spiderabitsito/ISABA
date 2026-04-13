"""**************** ESP32 CAM STREAM ****************
Servidor HTTP que envía frames JPEG
***************************************************"""

import network
import socket
from camera import Camera, FrameSize, PixelFormat

# -------- WIFI AP ----------
ap_if = network.WLAN(network.AP_IF)
ap_if.active(True)

ap_if.config(
    essid="ESP32-S312",
    password="12345678",
    authmode=network.AUTH_WPA2_PSK,
    max_clients=4,
    channel=1,
    hidden=False
)

print("\n--- Punt d'accés actiu ---")
print("IP:", ap_if.ifconfig())

# -------- CAMERA ----------
cam = Camera(frame_size=FrameSize.QVGA, pixel_format=PixelFormat.JPEG)
cam.init()

# -------- WEB ----------
def web_page():
    html = """<html>
    <head>
        <title>ESP32-CAM</title>
    </head>
    <body>
        <h1>Streaming ESP32-CAM</h1>
        <img src="/frame" width="320">
    </body>
    </html>"""
    return html

# -------- SERVER ----------
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(('', 80))
s.listen(5)

print("🌐 Servidor actiu")

# -------- LOOP ----------
while True:
    try:
        client, addr = s.accept()
        request = client.recv(1024).decode()
        path = request.split(' ')[1]

        if path == "/frame":
            frame = cam.capture()

            if frame:
                client.send(b'HTTP/1.1 200 OK\r\nContent-Type: image/jpeg\r\n\r\n')
                client.sendall(frame)
            else:
                client.send(b'HTTP/1.1 500 Error\r\n\r\n')

        else:
            client.send(b'HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n\r\n')
            client.sendall(web_page().encode())

    except Exception as e:
        print("Error:", e)

    finally:
        client.close()
