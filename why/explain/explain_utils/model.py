import random 
import string

def get_model_framework(model):
    model_class_str = str(type(model))
    if "torch" in model_class_str:
        return "pytorch"
    elif "tensorflow" in model_class_str or "keras" in model_class_str:
        return "tensorflow"
    else:
        return None

def generate_random_name(N):
    return "".join(
        random.choice(string.ascii_uppercase + string.digits) for _ in range(N)
    )
