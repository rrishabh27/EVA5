import matplotlib.pyplot as plt
import numpy as np

import data_loading.transform as dt

def imshow(img, mean, std):
    un_norm = dt.UnNormalize(mean, std)
    img = un_norm(img)
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

def plot_dataset_images(device, classes, data_loader, mean, std, num_of_images=10):
    '''
    plots train dataset images(max 20 images)
    '''
    cnt = 0
    fig = plt.figure(figsize=(32,32))
    for data, target in data_loader:
        # data, target = data.to(device), target.to(device)
        for index, label in enumerate(target):
            title = "{}".format(classes[label.item()])
            
            ax = fig.add_subplot(num_of_images/5 + 1, 5, cnt+1, xticks=[], yticks=[])
            ax.set_title(title)
            imshow(data[index].cpu(), mean, std)
        
            cnt += 1
            if(cnt==num_of_images):
                break
        if(cnt==num_of_images):
            break
