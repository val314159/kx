from keras.models import load_model
model = load_model('nm.h5')
print(model)
print(model.layers)
