import matplotlib.pyplot as plt

num_epochs = len(Train_Error[:, 0])
# Plotting with matplotlib
plt.figure(figsize=(10, 6))

# Plotting each optimizer's training error
plt.plot(range(1, num_epochs + 1), Train_Error[:, 0], label='SGD', marker='o')
plt.plot(range(1, num_epochs + 1), Train_Error[:, 1], label='Adagrad', marker='s')
plt.plot(range(1, num_epochs + 1), Train_Error[:, 2], label='Adam', marker='^')

# Adding labels and title
plt.xlabel('Epochs')
plt.ylabel('Training Error')
plt.title('Training Error vs. Epochs')

# Adding legend
plt.legend()

# Display the plot
plt.grid(True)
plt.tight_layout()
plt.show()
