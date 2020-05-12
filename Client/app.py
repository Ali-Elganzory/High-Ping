import pygame
import pygame_gui
import numpy as np

from network import Network
from vision import Instructor_Encoder

# These are all the variables that you might need to change to test
drawing_color = True # If true another window will draw the color image with the foreground elements removed
camera = False # Change to True to use camera instead of local file
file_name = 'video_2.mp4' # If using local file, location of the file
start_frame_file = 150 # If using local file, frame to start the video at
end_frame_file = 2000 # If using local file, frame to stop the video at

pygame.init()

pygame.display.set_caption('High Ping')
window_surface = pygame.display.set_mode((800, 600))

login_screen_background = pygame.Surface((800, 600))
login_screen_background.fill(pygame.Color('#343a40'))
room_screen_background = pygame.Surface((800, 600))
room_screen_background.fill(pygame.Color('#ffffff'))

login_screen_manager = pygame_gui.UIManager((800, 600), 'theme.json')
room_screen_manager = pygame_gui.UIManager((800, 600), 'theme.json')

#         Login screen UI         #
#   top padding
height_top = 220

high_ping_label = pygame_gui.elements.UILabel(relative_rect=pygame.Rect((300, height_top - 130), (200, 40)),
                                              text="High Ping",
                                              object_id="high_ping",
                                              manager=login_screen_manager)

welcome_label = pygame_gui.elements.UILabel(relative_rect=pygame.Rect((300, height_top - 90), (200, 40)),
                                            text="Welcome back!",
                                            object_id="welcome",
                                            manager=login_screen_manager)

name_label = pygame_gui.elements.UILabel(relative_rect=pygame.Rect((275, height_top - 5), (40, 20)),
                                         text="Name",
                                         manager=login_screen_manager)

name_text_field = pygame_gui.elements.UITextEntryLine(relative_rect=pygame.Rect((275, height_top + 15), (250, 35)),
                                                      manager=login_screen_manager)

room_label = pygame_gui.elements.UILabel(relative_rect=pygame.Rect((275, height_top + 60), (40, 20)),
                                         text="Room",
                                         manager=login_screen_manager)

room_text_field = pygame_gui.elements.UITextEntryLine(relative_rect=pygame.Rect((275, height_top + 80), (250, 35)),
                                                      manager=login_screen_manager)

enter_room_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((320, height_top + 140), (160, 35)),
                                                 text='Enter room',
                                                 manager=login_screen_manager)

create_room_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((320, height_top + 190), (160, 35)),
                                                  text='Create room',
                                                  object_id="create_room",
                                                  manager=login_screen_manager)

name_error_label = pygame_gui.elements.UILabel(relative_rect=pygame.Rect((530, height_top + 15), (150, 35)),
                                               text="",
                                               object_id="error_label",
                                               manager=login_screen_manager)

room_error_label = pygame_gui.elements.UILabel(relative_rect=pygame.Rect((530, height_top + 80), (175, 35)),
                                               text="",
                                               object_id="error_label",
                                               manager=login_screen_manager)

#         Room screen UI         #
room_title_label = pygame_gui.elements.UILabel(
    relative_rect=pygame.Rect((60, 15), (250, 22)),
    text="name",
    object_id="room_title",
    manager=room_screen_manager)

client_name_label = pygame_gui.elements.UILabel(
    relative_rect=pygame.Rect((125, 35), (120, 22)),
    text="room",
    object_id="client_name",
    manager=room_screen_manager)

logout_button = pygame_gui.elements.UIButton(
    relative_rect=pygame.Rect((10, 10), (50, 50)),
    text="",
    object_id="logout",
    manager=room_screen_manager)

WIDTH = int(640/1.2)
HEIGHT = int(360/1.2)

screen_array_ones = np.ones((WIDTH, HEIGHT, 3))
screen_update_surface = pygame.surfarray.make_surface(0 * np.ones((WIDTH, HEIGHT, 3)))


def draw_screen_update(screen_update,w,h):
    WIDTH = w
    HEIGHT = h
    global screen_update_surface
    screen_update_surface = pygame.surfarray.make_surface(screen_update.transpose(1,0,2))


#   Manager & background to be rendered
background = login_screen_background
manager = login_screen_manager

validated = False
instructor = False
IS = None


def run_app():
    global background, manager
    global validated, instructor, IS

    clock = pygame.time.Clock()
    is_running = True

    while is_running:
        time_delta = clock.tick(60) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                network.disconnect()
                is_running = False

            if event.type == pygame.USEREVENT:
                if event.user_type == pygame_gui.UI_BUTTON_PRESSED:
                    if event.ui_element == enter_room_button:
                        validate_login(name_text_field.text, room_text_field.text, False)
                    elif event.ui_element == create_room_button:
                        validate_login(name_text_field.text, room_text_field.text, True)
                    elif event.ui_element == logout_button:
                        network.leave_room(room_title_label.text)
                        background = login_screen_background
                        manager = login_screen_manager
                        validated = False
                        continue

            manager.process_events(event)

        manager.update(time_delta)
        if validated and instructor:
            if camera:
                IS.send_data_camera()
            else:
                IS.send_data_file(network)
        window_surface.blit(background, (0, 0))
        if validated:
            window_surface.blit(screen_update_surface, (50, 75))
        manager.draw_ui(window_surface)

        pygame.display.update()
    if camera:
        IS.end_camera()


def validate_login(name, room, create):
    name_error_label.set_text("")
    room_error_label.set_text("")

    if len(name_text_field.text) < 1:
        name_error_label.set_text("Please, enter your name")
        if len(room_text_field.text) < 1:
            room_error_label.set_text("Please, enter the room name")
    elif len(room_text_field.text) < 1:
        room_error_label.set_text("Please, enter the room name")
    else:
        def entered_room(response):
            global validated, instructor, IS

            if "created" in response:
                if response["created"]:
                    render_room_screen()
                    IS = Instructor_Encoder(2, 20, drawing_color)
                    if camera:
                        IS.start_camera_enc(network)
                    else:
                        IS.start_file_enc(file_name, start_frame_file, end_frame_file, network)
                    validated = True
                    instructor = True
                else:
                    room_error_label.set_text("Busy room, choose another name")
            elif "entered" in response:
                if response["entered"]:
                    render_room_screen()
                    validated = True
                else:
                    room_error_label.set_text("No room")
            else:
                print(f"[Client] response error: {response}")

        def render_room_screen():
            global background, manager, client_name_label, room_title_label
            room_title_label.set_text(room)
            client_name_label.set_text(name)
            background = room_screen_background
            manager = room_screen_manager

        if create:
            network.create_room(name, room, on_response=entered_room)
        else:
            network.enter_room(name, room, on_response=entered_room)


if __name__ == "__main__":
    network = Network(update_screen=draw_screen_update)
    run_app()
