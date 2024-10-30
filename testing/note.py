def __init__(self, base_path):
    self.base_path = base_path
    self.image_size = (640, 640)  # Better for weapon detection


def load_and_preprocess_image(self, filename, label, bbox):
    try:
        img = tf.io.read_file(filename)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, self.image_size)
        img = tf.cast(img, tf.float32) / 255.0
        
        # Enhanced augmentation for weapon detection
        if tf.random.uniform([]) > 0.5:
            img = tf.image.random_flip_left_right(img)
            img = tf.image.random_brightness(img, 0.2)
            img = tf.image.random_contrast(img, 0.8, 1.2)
            img = tf.image.random_saturation(img, 0.8, 1.2)  # For different lighting
            # Random rotation for different angles
            img = tf.image.rot90(img, k=tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
            
        return img, label, bbox
    except tf.errors.NotFoundError:
        tf.print(f"File not found: {filename}")
        return None, None, None

def post_process_detections(self, predictions, confidence_threshold=0.5, iou_threshold=0.5):
    class_pred, bbox_pred = predictions
    
    # Get predictions above threshold
    mask = np.max(class_pred, axis=1) > confidence_threshold
    filtered_boxes = bbox_pred[mask]
    filtered_scores = class_pred[mask]
    
    # Apply NMS
    selected_indices = tf.image.non_max_suppression(
        filtered_boxes,
        np.max(filtered_scores, axis=1),
        max_output_size=100,
        iou_threshold=iou_threshold
    )
    
    return filtered_boxes[selected_indices], filtered_scores[selected_indices]

def visualize_detection(image, boxes, scores, class_names):
    plt.figure(figsize=(12, 12))
    plt.imshow(image)
    
    for box, score in zip(boxes, scores):
        xmin, ymin, xmax, ymax = box
        class_id = np.argmax(score)
        confidence = np.max(score)
        
        # Denormalize coordinates
        xmin = int(xmin * self.image_size[1])
        ymin = int(ymin * self.image_size[0])
        xmax = int(xmax * self.image_size[1])
        ymax = int(ymax * self.image_size[0])
        
        # Draw box
        rect = plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin,
                           fill=False, color='red', linewidth=2)
        plt.gca().add_patch(rect)
        
        # Add label
        plt.text(xmin, ymin-10, 
                f'{class_names[class_id]}: {confidence:.2f}',
                color='red', fontsize=12)
    
    plt.axis('off')
    plt.show()

    def train_model():
    # ... existing code ...
    
    history = model.model.fit(
        train_dataset,
        epochs=100,
        validation_data=valid_dataset,
        callbacks=model.get_callbacks(),
        workers=1,
        use_multiprocessing=False,
        batch_size=8  # Adjusted for weapon detection
    )


# filename,class,xmin,ymin,xmax,ymax
# weapon1.jpg,handgun,100,150,300,400
# weapon2.jpg,rifle,200,250,400,500

def evaluate_model(model, test_dataset):
    # Compute mAP (mean Average Precision)
    mAP = tf.keras.metrics.MeanAveragePrecision()
    predictions = model.predict(test_dataset)
    mAP.update_state(test_dataset, predictions)
    print(f"mAP: {mAP.result().numpy()}")