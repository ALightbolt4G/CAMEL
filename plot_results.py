import matplotlib.pyplot as plt
import os

os.makedirs('paper/figures', exist_ok=True)

cells = ['history_cell', 'math_cell', 'code_cell']
start_loss = [3.327, 3.084, 2.722]
end_loss = [3.212, 2.883, 2.591]
accuracy = [0.3988, 0.4471, 0.4904]
time_mins = [47, 21, 12]

# 1. Loss Comparison
x = range(len(cells))
width = 0.35

fig, ax = plt.subplots()
rects1 = ax.bar([i - width/2 for i in x], start_loss, width, label='Start Loss', color='#ff9999')
rects2 = ax.bar([i + width/2 for i in x], end_loss, width, label='End Loss', color='#66b3ff')

ax.set_ylabel('Loss')
ax.set_title('Training Loss Comparison')
ax.set_xticks(x)
ax.set_xticklabels(cells)
ax.legend()
plt.savefig('paper/figures/loss_comparison.png')
plt.close()

# 2. Accuracy Comparison
plt.figure()
plt.bar(cells, accuracy, color='#99ff99')
plt.ylabel('Accuracy')
plt.title('Final Accuracy Comparison')
plt.savefig('paper/figures/accuracy_comparison.png')
plt.close()

# 3. Training Time
plt.figure()
plt.bar(cells, time_mins, color='#ffcc99')
plt.ylabel('Time (Minutes)')
plt.title('Training Time per Cell')
plt.savefig('paper/figures/training_time.png')
plt.close()

print("Figures generated successfully!")
