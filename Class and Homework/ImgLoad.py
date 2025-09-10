import matplotlib.pyplot as plt
import cv2

def main():
    img_path = '/home/kowsar/Documents/Image_Processing/DIP_Problems/img1.jpeg'
    img = cv2.imread(img_path)
    print('Image Shape : ', img.shape)

    print(img[:5, :5, 0])
    print(img.max(), img.min())

    # plot the image
    plt.imshow(img)
    plt.show()


if __name__ == '__main__':
    main()