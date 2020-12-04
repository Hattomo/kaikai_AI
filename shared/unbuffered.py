class Unbuffered:

    def __init__(self, stream, stringwriter):
        self.stream = stream
        self.stringwriter = stringwriter

    def write(self, data):
        self.stream.write(data)
        self.stream.flush()
        self.stringwriter.write(data)
        #f.write(data)

    def flush(self):
        pass