import pygame
import pygame_gui

from network import Network

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

#   Manager & background to be rendered
background = login_screen_background
manager = login_screen_manager


def run_app():
    global background, manager

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
                        continue

            manager.process_events(event)

        manager.update(time_delta)

        window_surface.blit(background, (0, 0))
        manager.draw_ui(window_surface)

        pygame.display.update()


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
            if "created" in response:
                if response["created"]:
                    render_room_screen()
                else:
                    room_error_label.set_text("Busy room, choose another name")
            elif "entered" in response:
                if response["entered"]:
                    render_room_screen()
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
    network = Network()
    run_app()
