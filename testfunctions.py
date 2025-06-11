import matplotlib.pyplot as plt
import torch

def calculate_l2_error(dataset, reference_img, l2_pre_registration, l2_post_registration, vis_every_n_img):
    
    for i in range(len(dataset)):
        mask, img_rgb, img_gray = dataset[i]
        cutout_img = mask * img_gray
        cutout_img = cutout_img.detach().numpy()
        moved, moving, fixed = register(moving=cutout_img, fixed=reference_img, anomaly=False)

        overlay = np.stack([moved.squeeze(0), fixed.squeeze(0), np.zeros_like(moved.squeeze(0))], axis=-1)

        l2_error_pre = np.linalg.norm(moving.squeeze(0) - fixed.squeeze(0))
        l2_error_post = np.linalg.norm(moved.squeeze(0) - fixed.squeeze(0))

        if l2_error_post/l2_error_pre > 0.85:
            print('Image Flipped')
            moved, moving, fixed = register(moving=cutout_img, fixed=reference_img, anomaly=True)
            l2_error_pre = np.linalg.norm(moving.squeeze(0) - fixed.squeeze(0))
            l2_error_post = np.linalg.norm(moved.squeeze(0) - fixed.squeeze(0))
            overlay = np.stack([moved.squeeze(0), fixed.squeeze(0), np.zeros_like(moved.squeeze(0))], axis=-1)
            
        l2_pre_registration.append(l2_error_pre)
        l2_post_registration.append(l2_error_post)


        if i%vis_every_n_img == 0:
            plt.imshow(overlay.squeeze(2))
            plt.show()
            print(f'L2 Error (absolute): Pre: {l2_error_pre} | Post: {l2_error_post}')
            print(f'L2 Error (relative): Pre: {l2_error_pre/(256*256)} | Post: {l2_error_post/(256*256)}')

    return l2_pre_registration, l2_post_registration
