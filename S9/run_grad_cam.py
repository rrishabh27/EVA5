import matplotlib.pyplot as plt
import seaborn as sns

import get_grad_cam, unnormalize

sns.set()


def plot_gradcam(model, device, test_loader, classes, target_layers):
        plt.style.use("dark_background")
        # logger.info('Plotting Grad-CAM...')

        # use the test images
        data, target = next(iter(test_loader))
        data, target = data.to(device), target.to(
            device)

        # logger.info('Taking {5} samples')
        # get 5 images
        data = data[:5]
        target = target[:5]

        # get the generated grad cam
        gcam_layers, predicted_probs, predicted_classes = get_grad_cam.get_grad_cam(device, model, 
        classes, data, target, target_layers)

        # get the denormalization function
        unorm = unnormalize.UnNormalize(mean, std)

        get_grad_cam.plot_gradcam(gcam_layers, data, target, predicted_classes,
                     classes, unorm)