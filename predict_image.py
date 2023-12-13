import torch
from torchvision import transforms, models
from PIL import Image


CLASS_NAMES = {
    "0": "apple_pie",
    "1": "baby_back_ribs",
    "2": "baklava",
    "3": "beef_carpaccio",
    "4": "beef_tartare",
    "5": "beet_salad",
    "6": "beignets",
    "7": "bibimbap",
    "8": "bread_pudding",
    "9": "breakfast_burrito",
    "10": "bruschetta",
    "11": "caesar_salad",
    "12": "cannoli",
    "13": "caprese_salad",
    "14": "carrot_cake",
    "15": "ceviche",
    "16": "cheesecake",
    "17": "cheese_plate",
    "18": "chicken_curry",
    "19": "chicken_quesadilla",
    "20": "chicken_wings",
    "21": "chocolate_cake",
    "22": "chocolate_mousse",
    "23": "churros",
    "24": "clam_chowder",
    "25": "club_sandwich",
    "26": "crab_cakes",
    "27": "creme_brulee",
    "28": "croque_madame",
    "29": "cup_cakes",
    "30": "deviled_eggs",
    "31": "donuts",
    "32": "dumplings",
    "33": "edamame",
    "34": "eggs_benedict",
    "35": "escargots",
    "36": "falafel",
    "37": "filet_mignon",
    "38": "fish_and_chips",
    "39": "foie_gras",
    "40": "french_fries",
    "41": "french_onion_soup",
    "42": "french_toast",
    "43": "fried_calamari",
    "44": "fried_rice",
    "45": "frozen_yogurt",
    "46": "garlic_bread",
    "47": "gnocchi",
    "48": "greek_salad",
    "49": "grilled_cheese_sandwich",
    "50": "grilled_salmon",
    "51": "guacamole",
    "52": "gyoza",
    "53": "hamburger",
    "54": "hot_and_sour_soup",
    "55": "hot_dog",
    "56": "huevos_rancheros",
    "57": "hummus",
    "58": "ice_cream",
    "59": "lasagna",
    "60": "lobster_bisque",
    "61": "lobster_roll_sandwich",
    "62": "macaroni_and_cheese",
    "63": "macarons",
    "64": "miso_soup",
    "65": "mussels",
    "66": "nachos",
    "67": "omelette",
    "68": "onion_rings",
    "69": "oysters",
    "70": "pad_thai",
    "71": "paella",
    "72": "pancakes",
    "73": "panna_cotta",
    "74": "peking_duck",
    "75": "pho",
    "76": "pizza",
    "77": "pork_chop",
    "78": "poutine",
    "79": "prime_rib",
    "80": "pulled_pork_sandwich",
    "81": "ramen",
    "82": "ravioli",
    "83": "red_velvet_cake",
    "84": "risotto",
    "85": "samosa",
    "86": "sashimi",
    "87": "scallops",
    "88": "seaweed_salad",
    "89": "shrimp_and_grits",
    "90": "spaghetti_bolognese",
    "91": "spaghetti_carbonara",
    "92": "spring_rolls",
    "93": "steak",
    "94": "strawberry_shortcake",
    "95": "sushi",
    "96": "tacos",
    "97": "takoyaki",
    "98": "tiramisu",
    "99": "tuna_tartare",
    "100": "waffles",
}


def get_model():
    return {
        "EfficientNetV2": models.efficientnet_v2_s(),
        "ResNet50": models.resnet50(),
        "VGG16": models.vgg16(),
        "MobileNetV3": models.mobilenet_v3_large(),
        "DenseNet121": models.densenet121(),
    }


def modify_model_classes(model_dict, num_classes):
    for name, model in model_dict.items():
        if name == "EfficientNetV2":
            num_ftrs = model.classifier[1].in_features
            model.classifier[1] = torch.nn.Linear(num_ftrs, num_classes)
        elif name == "ResNet50":
            num_ftrs = model.fc.in_features
            model.fc = torch.nn.Linear(num_ftrs, num_classes)
        elif name == "VGG16":
            num_ftrs = model.classifier[6].in_features
            model.classifier[6] = torch.nn.Linear(num_ftrs, num_classes)
        elif name == "MobileNetV3":
            num_ftrs = model.classifier[3].in_features
            model.classifier[3] = torch.nn.Linear(num_ftrs, num_classes)
        elif name == "DenseNet121":
            num_ftrs = model.classifier.in_features
            model.classifier = torch.nn.Linear(num_ftrs, num_classes)
    return model_dict


def predict_image(model_name, model_path, image_path):
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model
    model = modify_model_classes(get_model(), num_classes=101)[model_name]
    model.load_state_dict(torch.load(model_path, map_location=device))

    # Set the model to evaluation mode
    model.eval()
    model.to(device)

    # Load and transform the image
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Add batch dimension
    image = image.to(device)

    # Perform the prediction
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    return CLASS_NAMES[predicted.item()]


if __name__ == "__main__":
    model_name = "model name from dictionary above"
    model_path = "path to the model file"
    image_path = "path to the image file"

    prediction = predict_image(model_name, model_path, image_path)

    print(f"Predicted class: {prediction}")
