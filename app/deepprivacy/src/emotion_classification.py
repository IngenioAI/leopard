import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification


def test_emotion_class(img, fake_img):
    processor = AutoImageProcessor.from_pretrained("RickyIG/emotion_face_image_classification")
    model = AutoModelForImageClassification.from_pretrained("RickyIG/emotion_face_image_classification")
    model.eval()

    # original
    # img = Image.open(img_path).convert("RGB")
    inputs = processor(images=img, return_tensors="pt")

    # deepfake
    # fake_img = Image.open(fake_img_path).convert("RGB")
    fake_img_inputs = processor(images=fake_img, return_tensors="pt")

    with torch.no_grad():
        # original
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()

        # deepfake
        fake_img_outputs = model(**fake_img_inputs)
        logits2 = fake_img_outputs.logits
        fake_img_predicted_class_idx = logits2.argmax(-1).item()

    id2label = model.config.id2label
    predicted_label = id2label[predicted_class_idx]
    deepfake_predicted_label = id2label[fake_img_predicted_class_idx]

    return predicted_label, deepfake_predicted_label
