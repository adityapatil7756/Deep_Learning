import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from modules.image_classification import get_cifar10_data, build_simple_cnn, build_transfer_learning_model, train_and_evaluate
from modules.text_classification import get_imdb_data, build_rnn_model, build_lstm_model, build_gru_model, train_and_evaluate_text
from modules.explainability import get_gradcam_heatmap, explain_with_lime_image, explain_with_shap_image
from modules.fairness import introduce_bias, calculate_fairness_metrics, generate_counterfactual_image
from utils.visualization import plot_history, display_gradcam, plot_comparison
from utils.optimization import compare_optimizers, compare_initializations

# Set seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def run_image_module():
    print("\n--- Image Classification Module (Using Fashion MNIST) ---")
    (x_train, y_train), (x_test, y_test) = get_cifar10_data()
    
    # Use subset for faster demo
    x_train_sub, y_train_sub = x_train[:5000], y_train[:5000]
    x_test_sub, y_test_sub = x_test[:1000], y_test[:1000]
    
    print("Training Simple CNN (28x28x1)...")
    cnn_model = build_simple_cnn()
    cnn_hist, cnn_acc = train_and_evaluate(cnn_model, x_train_sub, y_train_sub, x_test_sub, y_test_sub, epochs=5)
    
    print("Preprocessing data for ResNet50 (32x32x3)...")
    # Resize to 32x32 and repeat channels to 3
    x_train_resnet = tf.image.resize(x_train_sub, (32, 32))
    x_train_resnet = tf.repeat(x_train_resnet, 3, axis=-1)
    x_test_resnet = tf.image.resize(x_test_sub, (32, 32))
    x_test_resnet = tf.repeat(x_test_resnet, 3, axis=-1)
    
    print("Training Transfer Learning Model (ResNet50)...")
    resnet_model = build_transfer_learning_model()
    resnet_hist, resnet_acc = train_and_evaluate(resnet_model, x_train_resnet, y_train_sub, x_test_resnet, y_test_sub, epochs=5)
    
    print(f"CNN Accuracy: {cnn_acc:.4f}")
    print(f"ResNet50 Accuracy: {resnet_acc:.4f}")
    
    plot_comparison({'Simple CNN': {'accuracy': cnn_acc}, 'ResNet50': {'accuracy': resnet_acc}})
    return cnn_model, resnet_model, (x_test_sub, y_test_sub)

def run_text_module():
    print("\n--- Text Classification Module ---")
    (x_train, y_train), (x_test, y_test) = get_imdb_data()
    
    # Subset
    x_train_sub, y_train_sub = x_train[:2000], y_train[:2000]
    x_test_sub, y_test_sub = x_test[:500], y_test[:500]
    
    models_to_train = {
        'RNN': build_rnn_model(),
        'LSTM': build_lstm_model(),
        'GRU': build_gru_model()
    }
    
    results = {}
    for name, model in models_to_train.items():
        print(f"Training {name}...")
        hist, acc = train_and_evaluate_text(model, x_train_sub, y_train_sub, x_test_sub, y_test_sub, epochs=3)
        results[name] = {'accuracy': acc}
        print(f"{name} Accuracy: {acc:.4f}")
        
    plot_comparison(results)

def run_explainability_demo(model, x_test):
    print("\n--- Explainability Module ---")
    img = x_test[0]
    
    # Grad-CAM
    print("Generating Grad-CAM heatmap...")
    # For our simple CNN, the last conv layer is likely 'conv2d_2' (check model.summary())
    try:
        last_conv_name = [l.name for l in model.layers if 'conv' in l.name][-1]
        heatmap = get_gradcam_heatmap(model, img[np.newaxis, ...], last_conv_name)
        gradcam_img = display_gradcam(img, heatmap)
        
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.title("Original Image")
        plt.subplot(1, 2, 2)
        plt.imshow(gradcam_img)
        plt.title("Grad-CAM")
        plt.show()
    except Exception as e:
        print(f"Grad-CAM error (likely layer name mismatch): {e}")

    # SHAP (Using a very small background for speed)
    print("Generating SHAP explanations...")
    try:
        background = x_test[1:20]
        shap_values = explain_with_shap_image(model, x_test[0:1], background)
        # Note: shap.image_plot handles its own plotting
        # shap.image_plot(shap_values, x_test[0:1])
        print("SHAP values computed.")
    except Exception as e:
        print(f"SHAP error: {e}")

def run_fairness_module(model, x_test, y_test):
    print("\n--- Fairness & Ethics Module ---")
    # Define a simple "protected group" based on brightness
    def is_bright(img): return np.mean(img) > 0.5
    
    protected_attr = np.array([is_bright(img) for img in x_test])
    y_pred = np.argmax(model.predict(x_test), axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    # Convert multiclass to binary for simplicity in metrics (e.g., class 0 vs others)
    y_true_bin = (y_true == 0).astype(int)
    y_pred_bin = (y_pred == 0).astype(int)
    
    metrics = calculate_fairness_metrics(y_true_bin, y_pred_bin, protected_attr)
    print(f"Fairness Metrics: {metrics}")
    
    # Counterfactual
    print("Generating Counterfactual image...")
    cf_img = generate_counterfactual_image(model, x_test[0], target_class=1)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(x_test[0])
    plt.title(f"Original (Pred: {y_pred[0]})")
    plt.subplot(1, 2, 2)
    plt.imshow(cf_img)
    plt.title(f"Counterfactual (Target Class: 1)")
    plt.show()

if __name__ == "__main__":
    cnn_model, resnet_model, (x_test, y_test) = run_image_module()
    run_text_module()
    run_explainability_demo(cnn_model, x_test)
    run_fairness_module(cnn_model, x_test, y_test)
    
    print("\n--- Final Insights ---")
    print("1. Transfer Learning (ResNet50) typically outperforms custom CNNs on CIFAR-10 but requires more resources.")
    print("2. LSTM and GRU usually converge faster and handle long-term dependencies better than standard RNNs.")
    print("3. Adam optimizer often converges faster than SGD, but SGD can sometimes generalize better.")
    print("4. Explainability tools like Grad-CAM and SHAP provide critical insights into 'why' a model makes a decision.")
    print("5. Fairness auditing reveals if a model is biased against specific data attributes.")
