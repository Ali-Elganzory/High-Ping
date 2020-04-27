import eventlet
import socketio

sio = socketio.Server()
app = socketio.WSGIApp(sio)

rooms = {}
instructors = {}
clients_in_rooms = {}


@sio.event
def connect(sid, environ):
    print("[Server] ", sid, "connected")


@sio.event
def create_room(sid, name, room):
    if room in rooms:
        print("[Server] ", sid, "failed to create", room)
        return {'created': False}
    else:
        instructors[sid] = room
        rooms[room] = {"instructor_id": sid, "instructor_name": name, "clients": {}}
        sio.enter_room(sid, room)
        print("[Server] ", sid, "created", room)
        return {'created': True}


@sio.event
def enter_room(sid, room, name):
    if room in rooms:
        rooms[room]["clients"][sid] = name
        clients_in_rooms[sid]["room"] = room
        clients_in_rooms[sid]["name"] = name
        sio.enter_room(sid, room)
        sio.emit('someone entered', f"{name} entered the room", room=room)
        print("[Server] ", sid, "entered", room)
        return {'entered': True, "instructor_name": rooms[room]["instructor_name"]}
    else:
        print("[Server] ", sid, "failed to enter", room)
        return {'entered': False}


@sio.event
def disconnect(sid):
    if sid in clients_in_rooms:
        sio.emit('someone left', f"{clients_in_rooms[sid]['name']} left the room", room=clients_in_rooms[sid]['room'])
        rooms[clients_in_rooms[sid]["room"]]["clients"].pop(sid)
        clients_in_rooms.pop(sid)

    print("[Server] ", sid, "disconnected")


if __name__ == "__main__":
    eventlet.wsgi.server(eventlet.listen(('', 5000)), app)

