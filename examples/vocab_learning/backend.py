class ChatBackend:
    def __init__(self, update_callback):
        # update_callback is a function to call when a new message is received
        self.update_callback = update_callback

    def send_message(self, message):
        # Here, you would implement the logic to send a message to the server
        # For now, we'll simulate a reply
        self.receive_message("Echo: " + message)

    def receive_message(self, message):
        # Call the update callback to update the UI with the new message
        if self.update_callback:
            self.update_callback(message)
