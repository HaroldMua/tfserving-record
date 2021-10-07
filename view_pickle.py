import pickle
f = open('object_detection/data/coco.pkl', 'rb+')
info = pickle.load(f)
print(info)
