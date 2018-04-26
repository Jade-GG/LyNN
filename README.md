# LyNN
This is a very simple C# neural network thing I made.
You can create networks as big as you want, and they can be saved to and loaded from disk.

Currently the default activation function is the ELU(Exponential Linear Unit), which is `y = x` for positive values and `y = a(e^x - 1)` for negative values.
Note that this can cause gradient explosion if you're not careful when training, or initialize the weights with extreme values.

Using only LyNN, I have been able to get 94% accuracy on the EMNIST digits dataset within 5 minutes of training on an i7 4770k(Using three hidden layers with 20 nodes each)

# Example
Setting up a network with a few layers, 4 inputs and 1 output, randomizing it, and training it with very basic data.

    //Building the network and randomizing it
    Network nw = Network.BuildNetwork(4, new int[]{10, 10, 10}, 1);
    nw.RandomizeNetwork(0.2f, 0.1f);
    
    //Feeding it two training examples
    nw.TrainNetwork(new float[] { 0.1526f, 0.1665f, 0.3283f, 0.3435f }, new float[]{ 0 });
    nw.TrainNetwork(new float[] { 0.4294f, 0.1469f, 0.0442f, 0.1560f }, new float[]{ 1 });

    //Applying the average nudges these two training examples gave
    nw.ApplyTrainingChanges(0.1f);


