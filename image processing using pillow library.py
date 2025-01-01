from PIL import Image
img = Image.open("/Users/yanagaur/Downloads/microbiome.jpg")
img2 = Image.open("/Users/yanagaur/Downloads/DetectIQ Letterhead Final 2.pdf/2.png")
print(type(img))
print(img.format)
print(img.size)
print(img.format)
small_img = img.resize((300,200))
small_img.save("/Users/yanagaur/Downloads/small.jpg")
img.thumbnail((300,250)) # only works on when the size is smaller and works with the aspect ratio
img.save("/Users/yanagaur/Downloads/smol.jpg")

cropped_img = img.crop((0,0,300,300))

cropped_img.save("/Users/yanagaur/Downloads/yana.jpg")

img_copy = img.copy()
img_copy.paste(img2,(50,50))
img_copy.save("/Users/yanagaur/Downloads/yana.jpg")
