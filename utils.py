import numpy as np
import math
import matplotlib.pyplot as plt
import torch
# WARMUP_METHOD ='exp'
LR_MAX=0.05
lr=0.05
N_EPOCHS= 100
N_WARMUP_EPOCHS = 10

def draw_plot(N_EPOCHS,learning_rates,BATCH_SIZE,iters,topk, topk_accuracy,losses,path_img):
    plt.title("Training Curve (batch_size={}, lr={})".format(BATCH_SIZE, lr))
    plt.plot(list(range(1,N_EPOCHS+1))  , learning_rates, label='Learning Rate')
    plt.xlabel('Iteration')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate over Iterations')
    plt.legend()
    plt.savefig(path_img + 'Learning_Rate.png')
    plt.show()
    
    for i, k in enumerate(topk):
        plt.plot(range(len(topk_accuracy)), [acc[i] for acc in topk_accuracy], label=f'top{k}')
    plt.title('Accuracy vs. Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(path_img + 'Learning_Rate.png')
    plt.show()

    plt.title("Losese Curve (batch_size={}, lr={})".format(BATCH_SIZE, lr))
    plt.plot(iters, losses, label="Train",color="red")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.savefig(path_img + 'Loss.png')
    plt.show()
    
# mnist_train_test = mnist_train[:100]
def lrfn(current_step, num_warmup_steps, lr_max,num_training_steps, num_cycles=0.50,WARMUP_METHOD='exp' ):
    if current_step < num_warmup_steps:
        if WARMUP_METHOD == 'log':
            return lr_max * 0.10 ** (num_warmup_steps - current_step)
        else:
            return lr_max * 2 ** -(num_warmup_steps - current_step)
    else:
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))

        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))) * lr_max
def plot_lr_schedule(lr_schedule, epochs):
    fig = plt.figure(figsize=(20, 10))
    plt.plot([None] + lr_schedule + [None])
    # X Labels
    x = np.arange(1, epochs + 1)
    x_axis_labels = [i if epochs <= 40 or i % 5 == 0 or i == 1 else None for i in range(1, epochs + 1)]
    plt.xlim([1, epochs])
    plt.xticks(x, x_axis_labels) # set tick step to 1 and let x axis start at 1

    # Increase y-limit for better readability
    plt.ylim([0, max(lr_schedule) * 1.1])

    # Title
    schedule_info = f'start: {lr_schedule[0]:.1E}, max: {max(lr_schedule):.1E}, final: {lr_schedule[-1]:.1E}'
    plt.title(f'Step Learning Rate Schedule, {schedule_info}', size=18, pad=12)

    # Plot Learning Rates
    for x, val in enumerate(lr_schedule):
        if epochs <= 40 or x % 5 == 0 or x is epochs - 1:
            if x < len(lr_schedule) - 1:
                if lr_schedule[x - 1] < val:
                    ha = 'right'
                else:
                    ha = 'left'
            elif x == 0:
                ha = 'right'
            else:
                ha = 'left'
            plt.plot(x + 1, val, 'o', color='black');
            offset_y = (max(lr_schedule) - min(lr_schedule)) * 0.02
            plt.annotate(f'{val:.1E}', xy=(x + 1, val + offset_y), size=12, ha=ha)

    plt.xlabel('Epoch', size=16, labelpad=5)
    plt.ylabel('Learning Rate', size=16, labelpad=5)
    plt.grid()
    plt.show()
def get_accuracy(model, data):
    correct = 0
    total = 0
    with torch.no_grad():
        model.eval()
        for inputs, labels in torch.utils.data.DataLoader(data, batch_size=len(data)):
            inputs, labels = inputs, labels = inputs.to(device), labels.to(device)
            output = model(inputs) # We don't need to run F.softmax
            # loss = criterion(inputs, labels)
            # val_loss += loss.item()
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(labels.view_as(pred)).sum().item()
            total += inputs.shape[0]
        model.train()
        # average_loss = val_loss / len(dataloader)
    return correct / total
if __name__ == '__main__':
    LR_SCHEDULE = [lrfn(step, num_warmup_steps=N_WARMUP_EPOCHS, lr_max=LR_MAX, num_training_steps=N_EPOCHS, num_cycles=0.50,) for step in range(N_EPOCHS)]
    plot_lr_schedule(LR_SCHEDULE, N_EPOCHS)
