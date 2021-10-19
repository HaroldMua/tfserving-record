import pickle
f = open('backend/object_detection/data/coco.pkl', 'rb+')
info = pickle.load(f)
print(info)
