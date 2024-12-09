def map_diagnos(benign_malignant, diagnosis):
    diag = 0
    match diagnosis:
        case 'melanoma': diag = 1 << 0
        case 'squamous cell carcinoma': diag = 1 << 1
        case 'basal cell carcinoma': diag = 1 << 2
        case 'pigmented benign keratosis': diag = 1 << 3
        case 'actinic keratosis': diag = 1 << 4
        case 'nevus': diag = 1 << 5
        case 'dermatofibroma': diag = 1 << 6
        case 'vascular lesion': diag = 1 << 7
        case 'seborrheic keratosis': diag = 1 << 8
        case 'solar lentigo': diag = 1 << 9
    if benign_malignant == 'malignant':
        diag = diag | (1 << 10)
    return diag

def map_diagnos_list(arr):
    return [map_diagnos(x[0], x[1]) for x in arr]