import pickle
f = open('../inference_client/object_detection/data/coco.pkl', 'rb+')
info = pickle.load(f)
print(info)
