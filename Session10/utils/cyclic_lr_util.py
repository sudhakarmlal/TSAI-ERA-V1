import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
def triangle_lr_plot(lr_min, lr_max, step_size, iterations):
    lr_list = []
    it_list = [j for j in range(iterations + 1)]
    for i in range(iterations + 1):
        half_cycle_count = np.floor(i/step_size)
        x = i - half_cycle_count * step_size
        if half_cycle_count % 2 == 0:
            lr = lr_min + x*(lr_max-lr_min)/step_size
        else:
            lr = lr_max - x*(lr_max-lr_min)/step_size
        lr_list.append(lr)
    fig, axs = plt.subplots(figsize=(5,5))
    axs.plot(it_list, lr_list)
    plt.title("CLR - 'triangular' Policy")
    plt.xlabel("Iteration")
    plt.ylabel("Learning Rate")
    plt.show()

def custom_one_cycle_lr(no_of_images, batch_size, base_lr, max_lr, final_lr, epoch_stage1=5, epoch_stage2=18, total_epochs=24):
    lr_schedule = lambda t: np.interp([t], [0, epoch_stage1, epoch_stage2, total_epochs], [base_lr,  max_lr, base_lr, final_lr])[0]
    lr_lambda = lambda it: lr_schedule(it * batch_size/no_of_images)
    
    return lr_lambda
	
def max_lr_finder_schedule(no_of_images, batch_size, base_lr, max_lr, total_epochs=5):
    lr_finder_schedule = lambda t: np.interp([t], [0, total_epochs], [base_lr,  max_lr])[0]
    lr_finder_lambda = lambda it: lr_finder_schedule(it * batch_size/no_of_images)
    
    return lr_finder_lambda