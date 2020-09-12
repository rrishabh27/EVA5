# misclassified = {1:[], 2:[], 3:[], 4:[], 5:[]}
# misclassified_pred = {1:[], 2:[], 3:[], 4:[], 5:[]}
# misclassified_target = {1:[], 2:[], 3:[], 4:[], 5:[]}
# misclassification_index = 1

import torch
import matplotlib.pyplot as plt

def get_misclassified(net):
    '''
    To get the misclassified images and their predicted and original values during the testing.
    '''
    misclassified = []
    misclassified_pred = []
    misclassified_target = []

    # put the model to evaluation mode
    net.eval()
    # turn off gradients
    with torch.no_grad():
        for data, target in test_loader:
            # move them to the respective device
            data, target = data.to(device), target.to(device)
            # do inferencing
            output = net(data)
            # get the predicted output
            pred = output.argmax(dim=1, keepdim=True)

            # get the current misclassified in this batch
            list_misclassified = (pred.eq(target.view_as(pred)) == False)
            batch_misclassified = data[list_misclassified]
            batch_mis_pred = pred[list_misclassified]
            batch_mis_target = target.view_as(pred)[list_misclassified]

            # batch_misclassified =

            misclassified.append(batch_misclassified)
            misclassified_pred.append(batch_mis_pred)
            misclassified_target.append(batch_mis_target)

    # group all the batched together
    misclassified = torch.cat(misclassified)
    misclassified_pred = torch.cat(misclassified_pred)
    misclassified_target = torch.cat(misclassified_target)

    return list(map(lambda x, y, z: (x, y, z), misclassified, misclassified_pred, misclassified_target))


def plot_misclassification(misclassified):
    '''
    To plot the misclassified images during testing of the model.
    '''

    # print('Total Misclassifications : {}'.format(len(misclassified)))
    num_images = 25
    fig = plt.figure(figsize=(12, 14))
    fig.suptitle('Misclassifications')
    for idx, (image, pred, target) in enumerate(misclassified[:num_images]):
        image, pred, target = image.cpu().numpy(), pred.cpu(), target.cpu() # converting the image back into numpy array for output
        ax = fig.add_subplot(5, 5, idx+1)
        ax.axis('off')
        ax.set_title('target {}\n pred {}'.format(target.item(), pred.item()), fontsize=16)
        ax.imshow(image.squeeze())
    plt.show()