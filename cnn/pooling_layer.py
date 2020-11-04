class Pooling_layer:
    def pooling(data,pooling_method,pooling_size):
        width = len(data)
        height = len(data[0])
        if pooling_method == "max-pooling":
            pass
        elif pooling_method == "mean-pooling":
            pass
        else:
            sys.stdout.write("Error: The pooling_method is not found\n")
            sys.exit(1)