# Collect images and labels from the existing iterator
def extract_data(data_iterator):
    images_list = []
    labels_list = []

    # Iterate through the entire dataset
    for i in range(len(data_iterator)):
        batch_images, batch_labels = data_iterator[i]
        images_list.append(batch_images)
        labels_list.append(batch_labels)

    # Stack the batches into NumPy arrays
    images_array = np.vstack(images_list)
    labels_array = np.vstack(labels_list)
    
    return images_array, labels_array

# Extract images and labels
train_images, train_labels = extract_data(train_data)
val_images, val_labels = extract_data(val_data)
test_images, test_labels = extract_data(test_data)

# Check whether the new sets are the right shape
print("Train set:", train_images.shape, train_labels.shape)
print("Validation set:", val_images.shape, val_labels.shape)
print("Test set:", test_images.shape, test_labels.shape)

def denoise(img):
    sigma_est = estimate_sigma(img)
    img = denoise_nl_means(img, h=1.*sigma_est, fast_mode=False, 
                                      patch_size=5, patch_distance=3)
    return img

data_gen_aug = ImageDataGenerator(preprocessing_function=denoise)  # Example: without noise or with different augmentation

train_data_aug = data_gen_aug.flow(
    train_images, train_labels,
    batch_size=batch_size,
    shuffle=False  # You can control whether to shuffle here
)

val_data_aug = data_gen_aug.flow(
    val_images, val_labels,
    batch_size=batch_size,
    shuffle=False
)

test_data_aug = data_gen_aug.flow(
    test_images, test_labels,
    batch_size=batch_size,
    shuffle=False
)