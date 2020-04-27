import pygame
import pygame_gui

from network import Network

pygame.init()

pygame.display.set_caption('Quick Start')
window_surface = pygame.display.set_mode((800, 600))

background = pygame.Surface((800, 600))
background.fill(pygame.Color('#343a40'))

manager = pygame_gui.UIManager((800, 600), 'theme.json')

#   top padding
height_top = 220

high_ping_label = pygame_gui.elements.UILabel(relative_rect=pygame.Rect((300, height_top - 130), (200, 40)),
                                              text="High Ping",
                                              object_id="high_ping",
                                              manager=manager)

welcome_label = pygame_gui.elements.UILabel(relative_rect=pygame.Rect((300, height_top - 90), (200, 40)),
                                            text="Welcome back!",
                                            object_id="welcome",
                                            manager=manager)

name_label = pygame_gui.elements.UILabel(relative_rect=pygame.Rect((275, height_top - 5), (40, 20)),
                                         text="Name",
                                         manager=manager)

name_text_field = pygame_gui.elements.UITextEntryLine(relative_rect=pygame.Rect((275, height_top + 15), (250, 35)),
                                                      manager=manager)

room_label = pygame_gui.elements.UILabel(relative_rect=pygame.Rect((275, height_top+60), (40, 20)),
                                         text="Room",
                                         manager=manager)

room_text_field = pygame_gui.elements.UITextEntryLine(relative_rect=pygame.Rect((275, height_top + 80), (250, 35)),
                                                      manager=manager)

enter_room_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((320, height_top + 140), (160, 35)),
                                                 text='Enter room',
                                                 manager=manager)

create_room_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((320, height_top + 190), (160, 35)),
                                                  text='Create room',
                                                  object_id="create_room",
                                                  manager=manager)


def run_app():
    clock = pygame.time.Clock()
    is_running = True

    while is_running:
        time_delta = clock.tick(60) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                is_running = False

            if event.type == pygame.USEREVENT:
                if event.user_type == pygame_gui.UI_BUTTON_PRESSED:
                    if event.ui_element == enter_room_button:
                        print('enter room')
                    elif event.ui_element == create_room_button:
                        print('create room')

            manager.process_events(event)

        manager.update(time_delta)

        window_surface.blit(background, (0, 0))
        manager.draw_ui(window_surface)

        pygame.display.update()


if __name__ == "__main__":
    network = Network()
    run_app()

