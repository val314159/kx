from networks import mkFinetune
model = mkFinetune()

print("let's look at the layers...")
from pprint import pprint
pprint(model.layers)
z = model.layers[-1]
model.pop()
for lyr in z.layers:
    model.add(lyr)
pprint(model.layers)

model.save("nm.h5")
