import socketio

sio = socketio.Client()


class Network(object):
    def __init__(self):
        sio.register_namespace(MainNamespace('/'))
        sio.connect('http://localhost:5000')


class MainNamespace(socketio.ClientNamespace):
    def on_connect(self):
        sio.emit('create_room', ("Ali Sayed", "Software Engineering"), callback=self.check_create_room)

    def on_disconnect(self):
        pass

    def on_my_event(self, data):
        self.emit('my_response', data)

    @staticmethod
    def check_create_room(response):
        print(response["created"])

    @staticmethod
    def check_enter_room(response):
        print(response["created"])
