import matplotlib.pyplot as plt
import cv2

def main():
    img_path = '/home/kowsar/Documents/Image_Processing/DIP_Problems/img1.jpeg'
    img = cv2.imread(img_path)
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print('Image Shape : ', image.shape)

    c = 50
    new_img = image + c

    # Plot all images
    plt.figure(figsize=(12,10))

    plt.subplot(1,2,1)
    plt.title('Orginal Image')
    plt.imshow(image)

    plt.subplot(1,2,2)
    plt.title('Increased pixel value')
    plt.imshow(new_img)

    plt.show()


if __name__ == '__main__':
    main()    