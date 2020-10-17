import torch.optim as optim

import trainer, tester



def evaluation(net, train_loader, test_loader, optimizer, scheduler, epochs, device, train_acc, train_losses, test_acc, test_losses):

    # initialising cumulative train and test metrics which will store per epoch metrics
    # cum_train_acc = []
    # cum_train_losses = []
    # cum_test_acc = []
    # cum_test_losses = []

    # net = model.Cifar10_Net(norm_type = 'BN').to(device)
    # scheduler = StepLR(optimizer, step_size=6, gamma=0.1)


    for epoch in range(1, epochs+1):
        print('\n Epoch:', epoch)
        trainer.train(net, device, train_loader, optimizer, scheduler, epoch, train_acc, train_losses)
        # scheduler.step() # reduce lr on plateau
        tester.test(net, device, test_loader, test_acc, test_losses)

        # cum_train_acc.extend(train_acc)
        # cum_train_losses.extend(train_losses)
        # cum_test_acc.extend(test_acc)
        # cum_test_losses.extend(test_losses)

    # return (cum_train_acc, cum_train_losses, cum_test_acc, cum_test_losses)



