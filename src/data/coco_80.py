COCO_CLASSES = [
    "person", 
    "bicycle", 
    "car", 
    "motorcycle", 
    "airplane", 
    "bus", 
    "train", 
    "truck", 
    "boat", 
    "traffic light",
    "fire hydrant", 
    "stop sign", 
    "parking meter", 
    "bench", 
    "bird", 
    "cat", 
    "dog", 
    "horse", 
    "sheep", 
    "cow",
    "elephant", 
    "bear", 
    "zebra", 
    "giraffe", 
    "backpack", 
    "umbrella", 
    "handbag", 
    "tie", 
    "suitcase", 
    "frisbee",
    "skis", 
    "snowboard", 
    "sports ball", 
    "kite", 
    "baseball bat", 
    "baseball glove", 
    "skateboard", 
    "surfboard",
    "tennis racket", 
    "bottle", 
    "wine glass", 
    "cup", 
    "fork", 
    "knife", 
    "spoon", 
    "bowl", 
    "banana", 
    "apple",
    "sandwich", 
    "orange", 
    "broccoli", 
    "carrot", 
    "hot dog", 
    "pizza", 
    "donut", 
    "cake", 
    "chair", 
    "couch",
    "potted plant", 
    "bed", 
    "dining table", 
    "toilet", 
    "tv", 
    "laptop", 
    "mouse", 
    "remote", 
    "keyboard",
    "cell phone", 
    "microwave", 
    "oven", 
    "toaster", 
    "sink", 
    "refrigerator", 
    "book", 
    "clock", 
    "vase",
    "scissors", 
    "teddy bear", 
    "hair drier", 
    "toothbrush"
]

COCO_VEHICLE = {
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    4: "airplane",
    5: "bus",
    6: "train",
    7: "truck",
    8: "boat"
}

COCO_ANIMAL = {
    14: "bird",
    15: "cat",
    16: "dog",
    17: "horse",
    18: "sheep",
    19: "cow",
    20: "elephant",
    21: "bear",
    22: "zebra",
    23: "giraffe"
}

COCO_FOOD = {
    46: "banana",
    47: "apple",
    48: "sandwich",
    49: "orange",
    50: "broccoli",
    51: "carrot",
    52: "hot dog",
    53: "pizza",
    54: "donut",
    55: "cake",
    39: "bottle",
    40: "wine glass",
    41: "cup",
    42: "fork",
    43: "knife",
    44: "spoon",
    45: "bowl"
}

COCO_ELECTRONICS = {
    63: "laptop",
    64: "mouse",
    65: "remote",
    66: "keyboard",
    67: "cell phone",
    68: "microwave",
    69: "oven",
    70: "toaster",
    71: "sink",
    72: "refrigerator"
}

COCO_DAILY = {
    24: "backpack",
    25: "umbrella",
    26: "handbag",
    27: "tie",
    28: "suitcase",
    73: "book",
    74: "clock",
    75: "vase",
    76: "scissors",
    77: "teddy bear",
    78: "hair drier",
    79: "toothbrush"
}

COCO_ROAD = {
    9: "traffic light",
    10: "fire hydrant",
    11: "stop sign",
    12: "parking meter",
    13: "bench"
}

COCO_SPORTS = {
    29: "frisbee",
    30: "skis",
    31: "snowboard",
    32: "sports ball",
    33: "kite",
    34: "baseball bat",
    35: "baseball glove",
    36: "skateboard",
    37: "surfboard",
    38: "tennis racket"
}

COCO_PERSON = {
    0: "person"
}

COCO_GROUPS = {
    "person": COCO_PERSON,
    "vehicle": COCO_VEHICLE,
    "animal": COCO_ANIMAL,
    "food": COCO_FOOD,
    "electronics": COCO_ELECTRONICS,
    "daily": COCO_DAILY,
    "road": COCO_ROAD,
    "sports": COCO_SPORTS
}

def find_class_id(groups: tuple, id: int):
    
    if (id > len(COCO_CLASSES)):
        return str(id)
    
    if (groups is None):
        return COCO_CLASSES[id]
    
    
    name = ""
    for group in groups:
        coco_class = COCO_GROUPS.get(group)
        name = coco_class.get(id)
        
        if (name is not None):
            return name
        
    return None
    