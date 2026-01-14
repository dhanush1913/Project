import numpy as np

def fgsm_attack(image, gradient, epsilon=0.3):
    perturbation = epsilon * np.sign(gradient)
    adv_image = image + perturbation
    return np.clip(adv_image, 0, 1)

def pgd_attack(image, gradient_fn, epsilon=0.3, alpha=0.01, num_steps=40):
    adv_image = image.copy()
    
    for _ in range(num_steps):
        gradient = gradient_fn(adv_image)
        adv_image = adv_image + alpha * np.sign(gradient)
        perturbation = np.clip(adv_image - image, -epsilon, epsilon)
        adv_image = image + perturbation
        adv_image = np.clip(adv_image, 0, 1)
    
    return adv_image

def generate_random_adversarial(image, epsilon=0.3):
    perturbation = np.random.uniform(-epsilon, epsilon, image.shape)
    adv_image = image + perturbation
    return np.clip(adv_image, 0, 1)

def generate_fgsm_dataset(images, epsilon=0.3, use_random_gradient=True):
    adv_images = []
    
    for img in images:
        if use_random_gradient:
            gradient = np.random.randn(*img.shape)
        else:
            gradient = np.random.randn(*img.shape)
        adv_img = fgsm_attack(img, gradient, epsilon)
        adv_images.append(adv_img)
    
    return np.array(adv_images)

def generate_pgd_dataset(images, epsilon=0.3, alpha=0.01, num_steps=40, use_random_gradient=True):
    adv_images = []
    
    for img in images:
        if use_random_gradient:
            def gradient_fn(x):
                return np.random.randn(*x.shape) * 0.1  # Smaller for iterative
        else:
            def gradient_fn(x):
                return np.random.randn(*x.shape) * 0.1
        
        adv_img = pgd_attack(img, gradient_fn, epsilon, alpha, num_steps)
        adv_images.append(adv_img)
    
    return np.array(adv_images)
