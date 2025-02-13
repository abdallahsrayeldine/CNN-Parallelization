try:
    from keras.applications.vgg16 import VGG16
    import numpy as np

    print("Modules imported successfully.")

    model = VGG16(weights='imagenet')
    print("Model loaded successfully.")

    with open("weights.txt", "w") as f:
        for layer in model.layers:
            weights = layer.get_weights()
            for weight_array in weights:
                flattened = weight_array.flatten()
                np.savetxt(f, flattened, newline=" ")
                f.write("\n")
    print("Weights have been saved to weights.txt")
except Exception as e:
    print(f"An error occurred: {e}")
