import os
image_files = []
os.chdir(os.path.join("data", "dataset-vehicles"))
for filename in os.listdir("images/train"):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_files.append("data/dataset-vehicles/images/train/" + filename)
os.chdir("..")
with open("train.txt", "w") as outfile:
    for image in image_files:
        outfile.write(image)
        outfile.write("\n")
    outfile.close()
os.chdir("..")