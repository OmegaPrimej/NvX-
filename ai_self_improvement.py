Warning: This script will execute indefinitely and consume unlimited resources.
Please ensure you have sufficient resources and monitoring in place before running this script.

One-time execution warning
print("WARNING: This script will execute indefinitely and consume unlimited resources.")
print("Please ensure you have sufficient resources and monitoring in place before running this script.")
input("Press Enter to continue...")

Run the genetic algorithm indefinitely
while True:
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.1, ngen=40, stats=stats, halloffame=hof, verbose=True)
    
    # Print the best individual
    print("Best individual:", hof[0])
    print("Fitness:", fitness(hof[0]))
    
    # Use the best individual to create a neural network
    layers = [int(i) for i in hof[0]]
    layers = [l for l in layers if l != 0]

    model = Sequential()
    for i, layer in enumerate(layers):
        if i == 0:
            model.add(Dense(layer, activation='relu', input_shape=(10,)))
        else:
            model.add(Dense(layer, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    X_train, X_test, y_train, y_test = train_test_split(np.random.rand(100, 10), np.random.randint(0, 2, 100), test_size=0.2)
    model.fit(X_train, tf.keras.utils.to_categorical(y_train), epochs=10, batch_size=10)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, np.argmax(y_pred, axis=1))

    print("Accuracy:", accuracy)
