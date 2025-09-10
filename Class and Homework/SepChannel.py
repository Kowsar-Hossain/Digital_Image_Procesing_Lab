import matplotlib.pyplot as plt
import cv2

def main():
    img_path = '/home/kowsar/Documents/Image_Processing/DIP_Problems/img1.jpeg'
    image = cv2.imread(img_path)
    print("Image shape : ", image.shape)

    # Convert to BGR to RCB
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # plot the image in Separate Channel
    plt.figure(figsize=(10,10))

    plt.subplot(2,3,1)
    plt.imshow(img_rgb)
    plt.title('Original Image')

    plt.subplot(2,3,2)
    plt.imshow(img_rgb[:,:,0], cmap='Reds')
    plt.title('Red Channel')

    plt.subplot(2,3,3)
    plt.imshow(img_rgb[:,:,1], cmap='Greens')
    plt.title('Green Channel')

    plt.subplot(2,3,4)
    plt.imshow(img_rgb[:,:,2], cmap='Blues')
    plt.title('Blues Channel')

    plt.subplot(2,3,5)
    plt.imshow(img_rgb[:,:,0], cmap='gray')
    plt.title('Gray Channel')

    plt.tight_layout()
    plt.show()
    
if __name__ == '__main__':
    main()
    