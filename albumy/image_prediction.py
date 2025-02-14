from transformers import BeitImageProcessor, BeitForImageClassification
from PIL import Image


def classify_image_beit(filename):
    # Load local image
    image_path = f"uploads/{filename}"  # Construct file path
    image = Image.open(image_path)

    # Load processor and model
    processor = BeitImageProcessor.from_pretrained('microsoft/beit-base-patch16-224-pt22k-ft22k')
    model = BeitForImageClassification.from_pretrained('microsoft/beit-base-patch16-224-pt22k-ft22k')

    # Process image
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    # Get prediction
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    predicted_classes = model.config.id2label[predicted_class_idx]
    return predicted_classes
# Example usage:
# result = classify_image_beit("e99bd2f9e6bf40b3aa6973ddab0c79a7.jpeg")
# print("Predicted class:", result)