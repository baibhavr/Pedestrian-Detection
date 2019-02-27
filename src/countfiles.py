import glob

pos = 0
# for files in glob.glob("img/train/pos/*.*"):
#     pos +=1
#     if (pos%100 == 0):
#         print pos

files = glob.glob("img/train/neg/*.*")
print(files)
print ("Number of files - ",len(files))