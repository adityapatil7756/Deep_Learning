import numpy as np
import tensorflow as tf
import lime
from lime import lime_image, lime_text
import shap
import matplotlib.pyplot as plt

def get_gradcam_heatmap(model, img_array, last_conv_layer_name, pred_index=None):
    # Create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def explain_with_lime_image(model, image):
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(image.astype('double'), 
                                             model.predict, 
                                             top_labels=5, 
                                             hide_color=0, 
                                             num_samples=100)
    return explanation

def explain_with_lime_text(model, text_instance, class_names):
    explainer = lime_text.LimeTextExplainer(class_names=class_names)
    # LIME text explainer expects a function that takes a list of strings and returns probabilities
    def predict_proba(texts):
        # This is a placeholder; real implementation needs tokenization matching the model
        # We will handle this in the main script or a wrapper
        pass
    # For now, we define the structure
    return explainer

def explain_with_shap_image(model, images, background_images):
    # background_images should be a subset of training data
    explainer = shap.DeepExplainer(model, background_images)
    shap_values = explainer.shap_values(images)
    return shap_values

def explain_with_shap_text(model, x_test_subset, background_data):
    explainer = shap.DeepExplainer(model, background_data)
    shap_values = explainer.shap_values(x_test_subset)
    return shap_values
