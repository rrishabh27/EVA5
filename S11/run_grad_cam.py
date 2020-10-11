import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns

import get_grad_cam
import data_loading.transform as dt
# from utils import misclassifications

sns.set()


def plot_gradcam(model, device, test_loader, classes, target_layers, mean, std):
        plt.style.use("dark_background")

        # use the test images
        data, target = next(iter(test_loader))
        data, target = data.to(device), target.to(
            device)

        # get 5 images
        data = data[:5]
        target = target[:5]

        # get the generated grad cam
        gcam_layers, predicted_probs, predicted_classes = get_grad_cam.get_grad_cam(device, model, 
        classes, data, target, target_layers)

        # get the denormalization function
        unorm = dt.UnNormalize(mean, std)

        get_grad_cam.plot_gradcam(gcam_layers, data, target, predicted_classes,
                     classes, unorm)
        

def plot_gradcam_misclassified(model, test_loader, device, classes,
                             target_layers, mean, std, num_images = 25):
    misclassified = []
    misclassified_pred = []
    misclassified_target = []

    # put the model to evaluation mode
    model.eval()
    # turn off gradients
    with torch.no_grad():
        for data, target in test_loader:
            # move them to the respective device
            data, target = data.to(device), target.to(device)
            # do inferencing
            output = model(data)
            # get the predicted output
            pred = output.argmax(dim=1, keepdim=True)

            # get the current misclassified in this batch
            list_misclassified = (target.eq(pred.view_as(target)) == False)
            batch_misclassified = data[list_misclassified]
            batch_mis_pred = pred[list_misclassified]
            batch_mis_target = target[list_misclassified]

            # batch_misclassified =

            misclassified.append(batch_misclassified)
            misclassified_pred.append(batch_mis_pred)
            misclassified_target.append(batch_mis_target)

    # group all the batched together
    misclassified = torch.cat(misclassified)
    misclassified_pred = torch.cat(misclassified_pred)
    misclassified_target = torch.cat(misclassified_target)

    data = misclassified[:num_images]
    target = misclassified_target[:num_images]

    gcam_layers, predicted_probs, predicted_classes = get_grad_cam.get_grad_cam(device, model, 
        classes, data, target, target_layers)

    # get the denormalization function
    unorm = dt.UnNormalize(mean, std)

    # plot gradcam on misclassified images(using predicted_classes instead of pred)
    get_grad_cam.plot_gradcam(gcam_layers, data, target, predicted_classes, classes, unorm)



