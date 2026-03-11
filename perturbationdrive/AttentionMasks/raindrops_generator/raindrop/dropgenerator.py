import random
from random import randint
import cv2
import math
import numpy as np
from PIL import Image
from PIL import ImageEnhance
from PIL import ImageDraw
from skimage.measure import label as skimage_label
from .raindrop import Raindrop, make_bezier
from .snowflake import SnowFlake

"""
This module contain two functions:
Check Collision -- handle the collision of the drops
generateDrops -- generate raindrops on the image

Author: Chia-Tse, Chang
Edited by Vera, Soboleva
"""


def CheckCollision(DropList):
	"""
	This function handle the collision of the drops
	:param DropList: list of raindrop class objects 
	"""
	listFinalDrops = []
	Checked_list = []
	list_len = len(DropList)
	# because latter raindrops in raindrop list should has more colision information
	# so reverse list	
	DropList.reverse()
	drop_key = 1
	for drop in DropList:
		# if the drop has not been handle	
		if drop.getKey() not in Checked_list:			
			# if drop has collision with other drops
			if drop.getIfColli():
				# get collision list
				collision_list = drop.getCollisionList()
				# first get radius and center to decide how  will the collision do
				final_x = drop.getCenters()[0] * drop.getRadius()
				final_y = drop.getCenters()[1]  * drop.getRadius()
				tmp_devide = drop.getRadius()
				final_R = drop.getRadius()  * drop.getRadius()
				for col_id in collision_list:
					col_id = int(col_id)
					Checked_list.append(col_id)
					# list start from 0
					final_x += DropList[list_len - col_id].getRadius() * DropList[list_len - col_id].getCenters()[0]
					final_y += DropList[list_len - col_id].getRadius() * DropList[list_len - col_id].getCenters()[1]
					tmp_devide += DropList[list_len - col_id].getRadius()
					final_R += DropList[list_len - col_id].getRadius() * DropList[list_len - col_id].getRadius() 
				final_x = int(round(final_x/tmp_devide))
				final_y = int(round(final_y/tmp_devide))
				final_R = int(round(math.sqrt(final_R)))
				# rebuild drop after handled the collisions
				newDrop = Raindrop(drop_key, (final_x, final_y), final_R)
				drop_key = drop_key+1
				listFinalDrops.append(newDrop)
			# no collision
			else:
				drop.setKey(drop_key)
				drop_key = drop_key+1
				listFinalDrops.append(drop)
	

	return listFinalDrops

def generate_label(h, w, cfg):
    """
    This function generates a list of raindrop class objects and a label map of these drops in the image.
    :param h: image height
    :param w: image width
    :param cfg: config with global constants
	:return: list of final raindrops
    """

    maxDrop = cfg["maxDrops"]
    minDrop = cfg["minDrops"]
    maxR = cfg["maxR"]
    minR = cfg["minR"]
    drop_num = random.randint(minDrop, maxDrop)
    #we want raindrops also beyond edges
    imgh = h + 5
    imgw = w + 5
    ran_pos = [(int(random.random() * imgw), int(random.random() * imgh)) for _ in range(drop_num)]
    
    listRainDrops = []
    
    for key, pos in enumerate(ran_pos):
        radius = random.randint(minR, maxR)
        shape = random.randint(0, 2)
        key = key + 1
        drop = Raindrop(key, pos, radius, shape)
        listRainDrops.append(drop)

    # Collision logic is skipped in the repo for this, so we skip it too.
    return listRainDrops, None, None
    

def generateDrops(bg_img, cfg, listFinalDrops):
    PIL_bg_img = Image.fromarray(bg_img)
    imgh, imgw, _ = bg_img.shape

    for idx, drop in enumerate(listFinalDrops):
        (ix, iy) = drop.getCenters()
        radius = drop.getRadius()
        if radius <= 0: continue

        drop_tex_h, drop_tex_w = drop.tex_h, drop.tex_w
        center_y, center_x = drop.tex_center_y, drop.tex_center_x
        paste_x = ix - center_x
        paste_y = iy - center_y

        # --- Correct Background Preparation with PADDING ---
        img_y_start = max(0, iy - center_y)
        img_y_end   = min(imgh, iy + (drop_tex_h - center_y))
        img_x_start = max(0, ix - center_x)
        img_x_end   = min(imgw, ix + (drop_tex_w - center_x))

        tmp_bg_slice = bg_img[img_y_start:img_y_end, img_x_start:img_x_end, :]

        padded_bg = np.zeros((drop_tex_h, drop_tex_w, 3), dtype=np.uint8)

        paste_y_start = max(0, center_y - iy)
        paste_y_end   = paste_y_start + (img_y_end - img_y_start)
        paste_x_start = max(0, center_x - ix)
        paste_x_end   = paste_x_start + (img_x_end - img_x_start)
        
        if tmp_bg_slice.shape[0] > 0 and tmp_bg_slice.shape[1] > 0 and \
           (paste_y_end - paste_y_start) == tmp_bg_slice.shape[0] and \
           (paste_x_end - paste_x_start) == tmp_bg_slice.shape[1]:
            
            padded_bg[paste_y_start:paste_y_end, paste_x_start:paste_x_end, :] = tmp_bg_slice
        
        try:
            # updateTexture uses the repo's logic now
            drop.updateTexture(padded_bg)
        except Exception as e:
            print(f"Warning: Failed to update texture for drop {idx}. Error: {e}")
            continue

        output = drop.getTexture() 
        if output is None:
            continue

        if hasattr(drop, "motion_length"):
            ml = getattr(drop, "motion_length", 0)
            if ml > 0:
                tex = drop.getTexture()
                arr = np.array(tex)
                # simple vertical streak
                kernel = np.ones((ml, 1), dtype=np.float32) / ml
                for c in range(3):
                    arr[:, :, c] = cv2.filter2D(arr[:, :, c], -1, kernel)
                output = Image.fromarray(arr)
            else:
                output = drop.getTexture()
         
        edge_ratio = cfg.get("edge_darkratio")
        enhancer = ImageEnhance.Brightness(output)
        output = enhancer.enhance(edge_ratio)  
          
        PIL_bg_img.paste(output, (paste_x, paste_y), output)

    final_image = np.asarray(PIL_bg_img)
    return final_image