# LyNN
This is a very simple C# neural network thing I made.
You can create networks as big as you want, and they can be saved to and loaded from disk.

# Example
Setting up a network with a few layers, 4 inputs and 1 output, randomizing it, and training it with very basic data.

    //Building the network and randomizing it
    Network nw = Network.buildNetwork(4, new int[]{10, 10, 10}, 1);
    nw.randomizeNetwork();
    
    //Feeding it two training examples
    nw.trainNetwork(new float[] { 0.1526f, 0.1665f, 0.3283f, 0.3435f }, new float[]{ 0 });
    nw.trainNetwork(new float[] { 0.4294f, 0.1469f, 0.0442f, 0.1560f }, new float[]{ 1 });

    //Applying the average nudges these two training examples gave
    nw.applyTrainingChanges(1);


