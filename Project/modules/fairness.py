import numpy as np
from sklearn.metrics import confusion_matrix

def introduce_bias(x, y, protected_attr_fn, target_class, drop_rate=0.8):
    """
    Artificially introduces bias by dropping samples of target_class 
    from the group where protected_attr_fn(x) is True.
    """
    mask = []
    for i in range(len(x)):
        is_protected = protected_attr_fn(x[i])
        is_target = np.argmax(y[i]) == target_class if len(y[i].shape) > 0 else y[i] == target_class
        
        if is_protected and is_target:
            if np.random.rand() < drop_rate:
                mask.append(False)
            else:
                mask.append(True)
        else:
            mask.append(True)
            
    return x[mask], y[mask]

def calculate_fairness_metrics(y_true, y_pred, protected_attr):
    """
    protected_attr: boolean array indicating group membership.
    """
    # Group 0: Not protected, Group 1: Protected
    y_true_g0 = y_true[~protected_attr]
    y_pred_g0 = y_pred[~protected_attr]
    y_true_g1 = y_true[protected_attr]
    y_pred_g1 = y_pred[protected_attr]
    
    # Selection Rate (Demographic Parity)
    sr_g0 = np.mean(y_pred_g0)
    sr_g1 = np.mean(y_pred_g1)
    dp_diff = abs(sr_g0 - sr_g1)
    
    # True Positive Rate (Equality of Opportunity)
    def get_tpr(t, p):
        cm = confusion_matrix(t, p, labels=[0, 1])
        tp = cm[1, 1]
        fn = cm[1, 0]
        return tp / (tp + fn) if (tp + fn) > 0 else 0
    
    tpr_g0 = get_tpr(y_true_g0, y_pred_g0)
    tpr_g1 = get_tpr(y_true_g1, y_pred_g1)
    eo_diff = abs(tpr_g0 - tpr_g1)
    
    return {
        'demographic_parity_diff': dp_diff,
        'equality_of_opportunity_diff': eo_diff,
        'selection_rates': (sr_g0, sr_g1),
        'tprs': (tpr_g0, tpr_g1)
    }

def generate_counterfactual_image(model, image, target_class, max_iter=100, lr=0.01):
    """
    Simple gradient-based counterfactual generation for images.
    Finds the smallest perturbation to make the model predict target_class.
    """
    image_tensor = tf.convert_to_tensor(image[np.newaxis, ...], dtype=tf.float32)
    perturbation = tf.Variable(tf.zeros_like(image_tensor))
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    
    for _ in range(max_iter):
        with tf.GradientTape() as tape:
            perturbed_image = tf.clip_by_value(image_tensor + perturbation, 0, 1)
            prediction = model(perturbed_image)
            loss = tf.keras.losses.sparse_categorical_crossentropy([target_class], prediction)
            
        grads = tape.gradient(loss, perturbation)
        optimizer.apply_gradients([(grads, perturbation)])
        
        if np.argmax(prediction.numpy()[0]) == target_class:
            break
            
    return tf.clip_by_value(image_tensor + perturbation, 0, 1).numpy()[0]
