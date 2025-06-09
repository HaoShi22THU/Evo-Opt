# from skimage.metrics import structural_similarity as ssim
# import cv2
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# # 读图并转换为灰度
# img1 = cv2.imread('/mnt/temp/hshi/EvoPress/EvoPress/generated_samples/img_0_revise.jpg', cv2.IMREAD_GRAYSCALE)
# img2 = cv2.imread('/mnt/temp/hshi/EvoPress/EvoPress/generated_samples/WechatIMG566.jpg', cv2.IMREAD_GRAYSCALE)

# score, ssim_map = ssim(img1, img2, full=True)
# print(f"SSIM = {score:.4f}")


import torch, clip, PIL.Image as Image
model, preprocess = clip.load('ViT-B/32')
img  = preprocess(Image.open('/mnt/temp/hshi/EvoPress/EvoPress/generated_samples/img_1_fy_015.jpg')).unsqueeze(0).cuda()
text = clip.tokenize(['A close-up high-contrast photo of Sydney Opera House sitting next to Eiffel tower, under a blue night sky of roiling energy, exploding yellow stars, and radiating swirls of blue.']).cuda()
with torch.no_grad():
    vi = model.encode_image(img)
    vt = model.encode_text(text)
    score = torch.cosine_similarity(vi, vt).item()
print(f'CLIP sim = {score:.3f}')