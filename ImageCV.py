import cv2
import numpy as np
import tensorflow as tf
import keras


def run(filepath):
    timg = cv2.imread(filepath, 100)
    print([len(timg), len(timg[0])])
    final = cv2.cvtColor(timg, cv2.COLOR_BGR2GRAY)
    final2 = cv2.cvtColor(timg, cv2.COLOR_BGR2GRAY)
    img = timg
    '''
    # Determine average deviation of rgb values in each pixel
    deviation = 0
    for x in range(len(img)):
        for y in range(len(img[0])):
            deviation += (abs(int(img[x, y, 0]) - int(img[x, y, 1])) +
                          (abs(int(img[x, y, 1]) - int(img[x, y, 2]))) / 2)
    deviation = deviation / (len(img) * len(img[0]))
    print deviation

    # Use deviation to turn only background into black
    for x in range(len(img)):
        for y in range(len(img[0])):
            if ((abs(int(img[x, y, 0]) - int(img[x, y, 1])) < deviation)):
                if (abs(int(img[x, y, 1]) - int(img[x, y, 2])) < deviation):
                    img[x, y, 0] = 0
                    img[x, y, 1] = 0
                    img[x, y, 2] = 0

    '''

    # Determine average deviation of rgb values in each pixel
    deviation = 0
    for x in range(len(final)):
        for y in range(len(final[0])):
            deviation += abs(int(final[x, y]))
    deviation = deviation / (len(final) * len(final[0]))
    # grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # blur = cv2.blur(grayscale, (5, 5))
    # ret, thresh = cv2.threshold(grayscale, 160, 255, cv2.THRESH_BINARY)

    # Convert from rgb to hsv
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Create mask separating background from object
    lower = np.array([0, 0, 0])
    upper = np.array([200, 150, 130])
    mask_1 = cv2.inRange(hsv, lower, upper)

    ret, mask_1 = cv2.threshold(final, 100, 255, cv2.THRESH_BINARY_INV)

    kernel = np.ones((5, 5), np.uint8)
    dilation = cv2.dilate(mask_1, kernel, iterations=5)

    # Find contours in masked image
    img_cnt, contours, hierarchy = cv2.findContours(
        dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(img_cnt, contours, -1, (0, 0, 0), 10)

    # Draw only the largest rectangle boundary (which will be the image)
    data = []
    rectangle = []
    max = 0
    for cnt in contours:
        (x, y, w, h) = cv2.boundingRect(cnt)
        if w < len(final[0]) - 10:
            if h < (len(final) - 10):
                if w * h > 800:
                    max = w * h
                    rectangle = [x, y, w, h]
                    data.append(rectangle)
                    cv2.rectangle(final, (rectangle[0], rectangle[1]), (rectangle[0] +
                                                                        rectangle[2], rectangle[1] + rectangle[3]), (0, 0, 0), 1)

    # cv2.imshow("Hello", final)
    # cv2.waitKey(0)

    j = 0
    for item in data:
        max = 0
        index = 0
        for i in range(len(data) - j):
            if data[i][1] > max:
                max = data[i][1]
                index = i
        temp = data[len(data) - 1 - j]
        data[len(data) - 1 - j] = data[index]
        data[index] = temp
        j += 1

    finalfinal = []

    def inside(rect1, rectangles):
        x1 = rect1[0]
        x2 = rect1[0] + rect1[2]
        y1 = rect1[1]
        y2 = rect1[1] + rect1[3]
        for s in range(len(rectangles)):
            x3 = rectangle[s][0]
            x4 = rectangle[s][0] + rectangle[s][2]
            y3 = rectangle[s][1]
            y4 = rectangle[s][1] + rectangle[s][3]
            if (x1 > x3) & (x2 < x4) & (y4 > y2) & (y3 < y1):
                return True
        return False

    # Crop the image upon the rectangle

    for k in range(len(data)):
        newimg = final2[(data[k][1]):(data[k][1] + data[k][3]),
                        (data[k][0]):(data[k][0] + data[k][2])]

        new_img = cv2.cvtColor(newimg, cv2.COLOR_GRAY2BGR)
        newimg = cv2.cvtColor(new_img, cv2.COLOR_BGR2HSV)
        img2 = cv2.cvtColor(new_img, cv2.COLOR_BGR2HSV)

        deviation = 0
        for x in range(len(newimg)):
            for y in range(len(newimg[0])):
                deviation += (abs(int(newimg[x, y, 2])))
        deviation = deviation / (len(newimg) * len(newimg[0]))

        lower = np.array([0, 0, 0])
        upper = np.array([0, 0, deviation - 10])
        mask = cv2.inRange(newimg, lower, upper)

        kernel = np.ones((3, 3), np.uint8)
        dilation_1 = cv2.dilate(mask, kernel, iterations=2)

        new_img_cnt, contours, hierarchy = cv2.findContours(
            dilation_1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        rectangle = []
        max = 0
        for cnt in contours:
            (x, y, w, h) = cv2.boundingRect(cnt)
            if w < len(newimg[0]) - 30:
                if h < (len(newimg) - 30):
                    if (w * h > 75):
                        rectangle.append([x, y, w, h])
                        cv2.rectangle(img2, (x, y), (x + w, y + h), (200, 200, 100), 1)

        # if (k == 2):
        #   cv2.imshow("Test", img2)
        #    cv2.waitKey(0)
        j = 0
        for item in rectangle:
            max = 0
            index = 0
            for i in range(len(rectangle) - j):
                if rectangle[i][0] > max:
                    max = rectangle[i][0]
                    index = i
            temp = rectangle[len(rectangle) - 1 - j]
            rectangle[len(rectangle) - 1 - j] = rectangle[index]
            rectangle[index] = temp
            j += 1
        test = []

        for n in range(len(rectangle)):
            if inside(rectangle[n], rectangle) == False:
                test.append(rectangle[n])

        rectangle = test

        a = 0

        def createimage(index, arr):
            finalimg = new_img[(rectangle[a][1]):(rectangle[a][1] + rectangle[a][3]),
                               (rectangle[a][0]):(rectangle[a][0] + rectangle[a][2])]

            ratio = len(finalimg[0]) * 1.0 / (len(finalimg) * 1.0)

            finalimg = cv2.resize(finalimg, (min(int(round(ratio * 20)), 28), 20))

            deviation = 0
            for x in range(len(finalimg)):
                for y in range(len(finalimg[0])):
                    deviation += int(finalimg[x, y, 0])
            deviation = deviation / (len(finalimg) * len(finalimg[0]))

            finalimg = cv2.cvtColor(finalimg, cv2.COLOR_BGR2GRAY)
            ret, finalimg = cv2.threshold(finalimg, deviation - 30, 255, cv2.THRESH_BINARY_INV)
            finalimg = cv2.blur(finalimg, (2, 2))
            # Open blank white image
            blank_image = cv2.imread("/Users/erichu/Desktop/white2.jpg", 10)
            blank_image = cv2.cvtColor(blank_image, cv2.COLOR_BGR2GRAY)
            blank_image[:, :] = [0]
            blank_image = blank_image[0:28, 0:28]

            border_width = (28 - len(finalimg)) / 2
            border_height = (28 - len(finalimg[0])) / 2

            # Place cropped picture onto white canvas
            for x in range(len(finalimg)):
                for y in range(len(finalimg[0])):
                    blank_image[border_width + x, border_height + y] = finalimg[x, y]

            return blank_image

        for a in range(len(rectangle)):
            final_img = createimage(a, rectangle)
            cv2.imwrite("/Users/erichu/Downloads/test" + str(a) + ".jpg", final_img)

        mnist = keras.datasets.mnist
        (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

        # cv2.imshow("Hi", test_images[0])
        # cv2.waitKey(0)

        train_images = train_images / 255.0
        test_images = test_images / 255.0
        model = keras.Sequential([
            keras.layers.Flatten(input_shape=(28, 28)),
            keras.layers.Dense(256, activation=tf.nn.relu),
            keras.layers.Dense(128, activation=tf.nn.relu),
            keras.layers.Dense(64, activation=tf.nn.relu),
            keras.layers.Dense(10, activation=tf.nn.softmax)
        ])

        model.load_weights("/Users/erichu/training1.ckpt")

        '''
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        model_cp = keras.callbacks.ModelCheckpoint(
            "/Users/ashwinr/Downloads/MenloHacks/training1.ckpt", save_weights_only = True, verbose = 1)

        model.fit(train_images, train_labels, epochs = 15, callbacks = [model_cp])

        test_loss, test_acc = model.evaluate(test_images, test_labels)

        print('Test accuracy:', test_acc)
        '''

        final = ""
        for i in range(len(rectangle)):
            img = cv2.imread("/Users/erichu/Downloads/test" + str(i) + ".jpg", 10)
            arr = np.array(img)
            arr = arr / 255.0
            arr = arr.reshape(1, 28, 28)
            predictions_single = model.predict(arr)
            p = np.argmax(predictions_single)
            final += str(p)

        if final != '':
            finalfinal.append(final)

    finalfinalfinal = []
    for p in range(len(finalfinal)):
        if p > 0:
            if ((finalfinal[p - 1]) != (finalfinal[p])):
                finalfinalfinal.append(finalfinal[p])
        else:
            finalfinalfinal.append(finalfinal[p])
    return finalfinalfinal


reader = open("/Users/erichu/Desktop/TAuto/fastgrader/fastgrader/work.txt", "r")
exam_file = reader.read()
print exam_file
print "Hello"
reader1 = open("/Users/erichu/Desktop/TAuto/fastgrader/fastgrader/plswork.txt", "r")
key_file = reader1.read()
print key_file
# filepath = reader.read()
exam = run(exam_file)
key = run(key_file)


# print key
# print exam

points = 0
for item in exam:
    for u in key:
        if item == u:
            points += 1

writer = open("/Users/erichu/Desktop/TAuto/fastgrader/fastgrader/plswork.txt", "w")
print "Score: " + str(points * 1.0 / len(key))
writer.write("Score: " + str(points * 1.0 / len(key)))
writer.close()
