import matplotlib.pyplot as plt
import torch

def calculate_l2_error(dataset, reference_img, l2_pre_registration, l2_post_registration, vis_every_n_img):
    
    for i in range(len(dataset)):
        print(i)
        mask, img_rgb, img_gray = dataset[i]
        cutout_img = mask * img_gray
        cutout_img = cutout_img.detach().numpy()
        moved, moving, fixed = register(moving=cutout_img, fixed=reference_img, anomaly=False)

        overlay = np.stack([moved.squeeze(0), fixed.squeeze(0), np.zeros_like(moved.squeeze(0))], axis=-1)
        plt.imshow(overlay.squeeze(2))
        plt.show()
        l2_error_pre = np.linalg.norm(moving.squeeze(0) - fixed.squeeze(0))
        l2_error_post = np.linalg.norm(moved.squeeze(0) - fixed.squeeze(0))
        l2_pre_registration.append(l2_error_pre)
        l2_post_registration.append(l2_error_post)
        print(f'Pre: {l2_error_pre} | Post: {l2_error_post}')



    return l2_pre_registration, l2_post_registration

