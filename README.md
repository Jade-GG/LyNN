# LyNN
This is a very simple C# neural network thing I made.
You can create networks as big as you want, and they can be saved to and loaded from disk.

# Example
Setting up a network with a few layers, 4 inputs and 1 output, randomizing it, and training it with very basic data.

    //Building the network and randomizing it
    Network nw = Network.buildNetwork(4, new int[]{10, 10, 10}, 1);
    nw.randomizeNetwork();
    
    //Feeding it two training examples
    nw.trainNetwork(new float[] { 2.1526f,  -6.1665f,  8.0831f,   -0.34355f }, new float[]{ 0 });
    nw.trainNetwork(new float[] { -0.4294f, -0.14693f, 0.044265f, -0.15605f }, new float[]{ 1 });

    //Applying the average nudges these two training examples gave
    nw.applyTrainingChanges(1);


