import matplotlib.pyplot as plt
import cv2

def main():
    img_path1 = '/home/kowsar/Documents/Image_Processing/DIP_Problems/Images/img1.jpeg'
    img_path2 = '/home/kowsar/Documents/Image_Processing/DIP_Problems/Images/img2.jpeg'

    img_1 = cv2.imread(img_path1)
    img_2 = cv2.imread(img_path2)

    img1_rgb = cv2.cvtColor(img_1, cv2.COLOR_BGR2RGB)
    img2_rgb = cv2.cvtColor(img_2, cv2.COLOR_BGR2RGB)

    resized1 = cv2.resize(img1_rgb, (300,200))
    resized2 = cv2.resize(img2_rgb, (300,200))

    print('First Image : ',resized1.shape)
    print('Second Image : ',resized2.shape)

    add2Img = resized1 + resized2

    #Plot the image 
    plt.figure(figsize=(12,10))

    plt.subplot(1,3,1)
    plt.imshow(resized1)
    plt.title('Cat Image')

    plt.subplot(1,3,2)
    plt.imshow(resized2)
    plt.title('Car Image')

    plt.subplot(1,3,3)
    plt.imshow(add2Img)
    plt.title('Added Image')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()    
