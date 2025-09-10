import matplotlib.pyplot as plt
import cv2
import numpy as np

def main():
    img_path = '/home/kowsar/Documents/Image_Processing/DIP_Problems/lily.jpeg'
    img = cv2.imread(img_path)
    print('Image Shape : ', img.shape)

    #--- Perform operations
    processed_img = 255 - img  # Invert the image colors
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    #Save Image
    save_path = '/home/kowsar/Documents/Image_Processing/DIP_Problems/img1_saved.png'
    cv2.imwrite(save_path, rgb_img)

    # plot the image
    figure_path = '/home/kowsar/Documents/Image_Processing/DIP_Problems/figure1.png'
    plt.figure(figsize=(15, 8))

    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.subplot(1, 2, 2)
    plt.imshow(rgb_img)
    plt.savefig(figure_path)
    plt.show()
    plt.close()

    #-- save Matrix
    npz_file = '/home/kowsar/Documents/Image_Processing/DIP_Problems/img2.npz'
    np.savez(npz_file, img, processed_img, rgb_img)
    #np.savez_compressed(npz_file, img, processed_img, rgb_img)

    img_set = np.load(npz_file)
    print(len(img_set), img_set['arr_0'])


if __name__ == '__main__':
    main()