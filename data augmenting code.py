# Preprocessing images stored as names a.png, to f.png likewise and saving it to a new directory named 'new'

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

data_aug = ImageDataGenerator(rotation_range = 25, width_shift_range = 0.19, height_shift_range = 0.19, zoom_range = 0.15)

p = "a"
for j in range(ord(p),ord("g")):
    
    image = load_img("png files\\" + chr(j) + ".png", color_mode="grayscale")
    data = img_to_array(image)
    d = data.reshape((1,) + data.shape)
    
    i=0
    for img in data_aug.flow(d, batch_size=1, save_to_dir="new", save_prefix=chr(j), save_format="jpg"):
        i += 1
        if i > 20:
            break
