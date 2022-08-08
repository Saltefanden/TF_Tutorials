import tensorflow as tf
import matplotlib.pyplot as plt

f_mnist = tf.keras.datasets.fashion_mnist

(x_train, y_train), (x_test, y_test) = f_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


x_train, x_test = x_train/ 255.0, x_test/ 255.0

if False:
    for i in range(25): plt.subplot(5,5,i+1); plt.yticks([]); plt.xticks([]); plt.imshow(x_train[i], cmap=plt.cm.binary); plt.xlabel(class_names[y_train[i]])
    plt.savefig('lol.png')

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(units=128, activation='relu'),
    tf.keras.layers.Dense(units=10)
])


model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

model.fit(x_train, y_train, epochs=10)

model.evaluate(x_test, y_test)

probability_model = tf.keras.Sequential([
    model,
    tf.keras.layers.Softmax()
])

predictions = probability_model.predict(x_test)




