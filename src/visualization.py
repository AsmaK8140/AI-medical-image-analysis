import pickle
import matplotlib.pyplot as plt

# Load history
with open("outputs/history.pkl", "rb") as f:
    history = pickle.load(f)

# Plot Accuracy
plt.plot(history['accuracy'], label='Train Accuracy')
plt.plot(history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig("outputs/accuracy.png")
plt.show()

# Plot Loss
plt.plot(history['loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig("outputs/loss.png")
plt.show()