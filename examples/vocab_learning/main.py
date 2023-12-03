from kivy.app import App
import ui

class ChatApp(App):
    def build(self):
        return ui.UI()

if __name__ == '__main__':
    ChatApp().run()
