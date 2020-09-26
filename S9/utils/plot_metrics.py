import seaborn as sns
import matplotlib.pyplot as plt
sns.set()
plt.style.use("dark_background")

def metrics(train_metric, test_metric):
    (train_acc, train_losses) = train_metric
    (test_acc, test_losses) = test_metric

    fig, axs = plt.subplots(2,2,figsize=(14,10))
    fig.suptitle('Metrics')
    axs[0, 0].plot(train_losses)
    axs[0, 0].set_title("Training Loss")
    axs[1, 0].plot(train_acc)
    axs[1, 0].set_title("Training Accuracy")
    axs[0, 1].plot(test_losses)
    axs[0, 1].set_title("Test Loss")
    axs[1, 1].plot(test_acc)
    axs[1, 1].set_title("Test Accuracy")
    plt.show()
