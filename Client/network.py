import socketio

from vision import Client_Reconstructor

sio = socketio.Client()
CR = Client_Reconstructor()

class Network(object):
    def __init__(self):
        sio.register_namespace(MainNamespace('/'))
        try:
            sio.connect('http://localhost:5000')
        except:
            print("[Client] connection refused!")

    def connect(self):
        sio.connect('http://localhost:5000')

    def disconnect(self):
        sio.disconnect()

    def enter_room(self, name, room, on_response):
        sio.emit('enter_room', (name, room), callback=on_response)

    def create_room(self, name, room, on_response):
        sio.emit('create_room', (name, room), callback=on_response)

    def leave_room(self, room):
        sio.emit('leave_room', room)

    def send_screen_update(self, screen_update):
        sio.emit('screen_update', screen_update)


class MainNamespace(socketio.ClientNamespace):
    def on_connect(self):
        print("[Client] connected")

    def on_disconnect(self):
        print("[Client] disconnected")

    def on_someone_entered(self, data):
        print(data)

    def on_someone_left(self, data):
        print(data)

    def on_room_closed(self, data):
        print(data)

    def on_screen_update(self, screen_update):
        print("screen_update!")
        current_bin_frame = CR.receive_data(screen_update)
        # print(current_bin_frame)
        # Draw image on pygame
