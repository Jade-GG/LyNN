using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;

namespace LyNN
{
    public enum NodeType
    {
        input,
        node,
        output,
    }

    public class Node
    {
        public NodeType type;
        public float value;
        public List<Weight> parents;
        public List<Weight> children;
        public float bias;

        //error values
        public float error;
        public float nact;
        public float bc;
        public int bc_count;
    }

    public class Weight
    {
        public float value;
        public Node parent;
        public Node child;

        public float vc;
        public int vc_count;
    }

    public class Network
    {
        public int numInputs;
        public int numOutputs;
        public int numLayers;

        private List<Node> inputs;
        private List<Node> outputs;
        private List<List<Node>> allNodes;
        private Random rand = new Random();

        //Clipping values to try prevention of gradient explosion
        public float gradient_clipping = 0.2f;
        public float weight_clipping = 1.0f;
        public float activation_clipping = 2.0f;


        /// <summary>
        /// Builds a network with the given amount of inputs, outputs, and hidden layers
        /// </summary>
        /// <param name="numInputs">Amount of inputs the network should have</param>
        /// <param name="layers">An array containing the amount of hidden nodes per layer</param>
        /// <param name="numOutputs">Amount of outputs the network should have</param>
        /// <returns>Returns a Network object</returns>
        public static Network BuildNetwork(int numInputs, int[] layers, int numOutputs)
        {
            //Sanity checking arguments
            if (numInputs < 1) throw new Exception("Can't have less than one input");
            if (numOutputs < 1) throw new Exception("Can't have less than one output");

            //Create and initialize the network
            Network net = new Network();
            net.numInputs = numInputs;
            net.numOutputs = numOutputs;
            net.numLayers = layers.Length;
            net.inputs = new List<Node>();
            net.outputs = new List<Node>();
            net.allNodes = new List<List<Node>>();

            //Build the node structure
            List<Node> prevs = new List<Node>();
            for(int i = -1; i < layers.Length + 1; i++)
            {
                int nc;
                NodeType t;

                //Lazy method to build the network with inputs and outputs all as 'nodes'
                if (i == -1)
                {
                    nc = numInputs;
                    t = NodeType.input;
                }
                else if (i == layers.Length)
                {
                    nc = numOutputs;
                    t = NodeType.output;
                }
                else
                {
                    nc = layers[i];
                    t = NodeType.node;

                    //Sanity checking on amount of nodes in layer
                    if (nc < 1) throw new Exception("Can't have less than one node in a layer");
                }

                List<Node> news = new List<Node>();
                for(int j = 0; j < nc; j++)
                {
                    //Create a new node with the specified type
                    Node n = new Node();
                    n.type = t;
                    n.value = 0;
                    n.children = new List<Weight>();
                    n.parents = new List<Weight>();
                    n.bias = 0;

                    //Run through all of the nodes in the previous layers and connect them
                    if (prevs.Count > 0)
                    {
                        foreach (Node pn in prevs)
                        {
                            Weight w = new Weight();
                            w.child = n;
                            w.parent = pn;
                            w.value = 0;

                            pn.children.Add(w);
                            n.parents.Add(w);
                        }
                    }

                    //Add to the input/output lists
                    if (t == NodeType.input) net.inputs.Add(n);
                    if (t == NodeType.output) net.outputs.Add(n);

                    //Add to 'news' list just so that it can be used as a reference for the nodes in the next layer
                    news.Add(n);
                }

                //Add the new layer of nodes to the allNodes list
                net.allNodes.Add(news);

                prevs.Clear();
                prevs.AddRange(news);
            }

            return net;
        }

        /// <summary>
        /// Randomizes the weights and biases for the network
        /// </summary>
        /// <param name="mulbias">The amount to multipli the randomized biases with(if 1, the values will be between -0.5 and 0.5)</param>
        /// <param name="mulweight">The amount to multipli the randomized weights with(if 1, the values will be between -0.5 and 0.5)</param>
        public void RandomizeNetwork(float mulbias = 1, float mulweight = 1)
        {
            for(int i = 0; i < allNodes.Count; i++)
            {
                for (int j = 0; j < allNodes[i].Count; j++)
                {
                    //Iterate through all nodes and randomize their bias
                    Node n = (allNodes[i])[j];
                    n.bias = ((float)rand.NextDouble() - 0.5f) * mulbias;

                    //Then iterate through all of its children weights and randomize them
                    for (int k = 0; k < n.children.Count; k++)
                    {
                        Weight w = n.children[k];
                        w.value = ((float)rand.NextDouble() - 0.5f) * mulweight;
                    }
                }
            }
        }

        /// <summary>
        /// Saves the network to file
        /// </summary>
        /// <param name="name">The filename to write it to</param>
        public void SaveNetwork(string name)
        {
            string nt = "";
            for(int i = 0; i < allNodes.Count; i++)
            {
                List<Node> nn = allNodes[i];

                //Write the amount of nodes in this layer on one line
                nt += nn.Count + "\n";
                foreach(Node n in nn)
                {
                    //Then, for each node, write the bias on one line, and all of its children weights in order of the node's index
                    nt += n.bias + "\n";
                    if (n.children.Count == 0)
                    {
                        nt += ";";
                    }
                    else
                    {
                        foreach (Weight w in n.children)
                        {
                            nt += w.value + ";";
                        }
                    }
                    nt += "\n";
                }
                nt += "\n";
            }
            File.WriteAllText(name, nt.Substring(0, nt.Length - 1)); //The -1 is to get rid of the double newline that would otherwise always exist at the end of the file
        }

        /// <summary>
        /// Loads a network from a file
        /// </summary>
        /// <param name="name">The filename to read from</param>
        public static Network LoadNetwork(string name)
        {
            //Create and initialize the network
            Network net = new Network();
            net.inputs = new List<Node>();
            net.outputs = new List<Node>();
            net.allNodes = new List<List<Node>>();

            //Read file
            //The layers are split by a double newline
            string[] all = File.ReadAllText(name).Split(new string[] { "\n\n" }, StringSplitOptions.RemoveEmptyEntries);
            net.numLayers = all.Length - 2;

            for(int i = 0; i < all.Length; i++)
            {
                net.allNodes.Add(new List<Node>());
            }

            for(int layer = all.Length - 1; layer >= 0; layer--)
            {
                string[] lines = all[layer].Split(new char[] { '\n' }, StringSplitOptions.RemoveEmptyEntries);

                int num = int.Parse(lines[0]);

                //Add to numinputs/outputs vars
                if (layer == 0) net.numInputs = num;
                if (layer == all.Length - 1) net.numOutputs = num;

                for (int i = 0; i < num; i++)
                {
                    Node n = new Node();
                    n.children = new List<Weight>();
                    n.parents = new List<Weight>();

                    if (layer == 0)
                    {
                        //Input nodes get marked as input nodes and added to the list of input nodes
                        net.inputs.Add(n);
                        n.type = NodeType.input;
                    }
                    else if (layer == all.Length - 1)
                    {
                        //Output nodes get marked as output nodes and added to the list of output nodes
                        net.outputs.Add(n);
                        n.type = NodeType.output;
                    }
                    else n.type = NodeType.node; //Everything else is a regular node

                    //Add this node to the list of all nodes, then start reading the values out of the file
                    net.allNodes[layer].Add(n);
                    n.bias = float.Parse(lines[i * 2 + 1]);

                    if (i * 2 + 2 < lines.Length)
                    {
                        //Add all of the children weights in order
                        string[] ws = lines[i * 2 + 2].Split(new char[] { ';' }, StringSplitOptions.RemoveEmptyEntries);
                        for(int j = 0; j < ws.Length; j++)
                        {
                            Weight w = new Weight();
                            w.parent = n;
                            w.child = net.allNodes[layer + 1][j];
                            w.value = float.Parse(ws[j]);
                            n.children.Add(w);
                            w.child.parents.Add(w);
                        }
                    }
                }
            }

            return net;
        }

        /// <summary>
        /// Exponential-based sigmoid activation function
        /// </summary>
        float Sigmoid_e(float val) { return (float)(1f / (1f + Math.Exp(-val))); }

        /// <summary>
        /// ELU rectifier activation function
        /// </summary>
        float ELU(float val)
        {
            if (val > activation_clipping) return activation_clipping;
            if (val >= 0) return val;
            else return (float)(Math.Exp(val) - 1);
        }

        /// <summary>
        /// Derivative of the exponential-based sigmoid activation function
        /// </summary>
        float D_Sigmoid_e(float val) { return val * (1 - val); }

        /// <summary>
        /// Derivative of the ELU rectifier function
        /// </summary>
        float D_ELU(float val)
        {
            if (val > activation_clipping) return 0;
            if (val >= 0) return 1;
            else return (float)Math.Exp(val);
        }

        /// <summary>
        /// Evaluates what the network thinks about the inputs
        /// </summary>
        /// <param name="inputs">The inputs to evaluate</param>
        /// <returns>Returns the output of the network</returns>
        public float[] Evaluate(float[] inputs)
        {
            if (inputs.Length != numInputs) throw new Exception("Incorrect number of inputs given! Got " + inputs.Length.ToString() + ", expected " + numInputs.ToString());

            for (int i = 0; i < inputs.Length; i++) this.inputs[i].value = inputs[i];

            //Calculate node value for each node in order
            for (int i = 0; i < allNodes.Count; i++)
            {
                for (int j = 0; j < allNodes[i].Count; j++)
                {
                    Node n = (allNodes[i])[j];
                    n.value = GetNodeVal(n);
                }
            }

            //Read the output nodes and write them into a float array to return
            float[] ret = new float[numOutputs];
            for (int i = 0; i < outputs.Count; i++) ret[i] = outputs[i].value;

            return ret;
        }

        /// <summary>
        /// Takes in training data and sets the error values accordingly
        /// </summary>
        /// <param name="inputs">Training data inputs</param>
        /// <param name="goodOutputs">Training data outputs(that the network should strive to get right)</param>
        /// <returns>Returns the total error in the output, lower is better</returns>
        public float TrainNetwork(float[] inputs, float[] goodOutputs)
        {
            //Feed the inputs to the network and evaluate
            float[] rets = Evaluate(inputs);

            //Calculate the total error to return later
            float errorsum = 0;
            for (int i = 0; i < goodOutputs.Length; i++)
            {
                outputs[i].error = goodOutputs[i] - rets[i];
                errorsum += GetError(goodOutputs[i], rets[i]);
            }

            //Go through all output nodes, adjust the biases and set nact values.
            for (int j = 0; j < outputs.Count; j++)
            {
                BackPropOutputOne(outputs[j]);
            }

            //Go through all nodes backwards(from output to input) and adjust the weights based on the error values. Skip the output nodes
            for (int i = numLayers; i >= 0; i--)
            {
                List<Node> nodes = allNodes[i];
                for (int j = 0; j < nodes.Count; j++)
                {
                    BackPropOne(nodes[j]);
                }
            }

            return errorsum;
        }

        /// <summary>
        /// Applies the average of all changes that the last training set proposed
        /// </summary>
        /// <param name="rate">The rate at which to change</param>
        public void ApplyTrainingChanges(float rate)
        {
            for (int i = numLayers + 1; i >= 0; i--)
            {
                List<Node> nodes = allNodes[i];
                for (int j = 0; j < nodes.Count; j++)
                {
                    Node n = nodes[j];
                    for (int x = 0; x < n.children.Count; x++)
                    {
                        Weight cw = n.children[x];
                        float cwv = rate * (cw.vc / (float)cw.vc_count);

                        //Apply gradient clipping
                        if (cwv > gradient_clipping) cwv = gradient_clipping;
                        if (cwv < -gradient_clipping) cwv = -gradient_clipping;
                        cw.value += cwv;

                        //Apply weight clipping
                        if (cw.value > weight_clipping) cw.value = weight_clipping;
                        if (cw.value < -weight_clipping) cw.value = -weight_clipping;

                        cw.vc = 0;
                        cw.vc_count = 0;
                    }
                    float bcv = rate * ((n.bc * n.bias) / (float)n.bc_count);

                    //Apply gradient clipping
                    if (bcv > gradient_clipping) bcv = gradient_clipping;
                    if (bcv < -gradient_clipping) bcv = -gradient_clipping;
                    n.bias += bcv;

                    //Apply 'weight' clipping
                    if (n.bias > weight_clipping) n.bias = weight_clipping;
                    if (n.bias < -weight_clipping) n.bias = -weight_clipping;

                    n.bc = 0;
                    n.bc_count = 0;
                }
            }
        }

        /// <summary>
        /// Backpropagate this node based on its children's error values
        /// </summary>
        /// <param name="n">The node to calculate error values for</param>
        void BackPropOne(Node n)
        {
            float err_sum = 0;
            for (int i = 0; i < n.children.Count; i++)
            {
                Weight cw = n.children[i];
                float nact = cw.child.nact;

                //Adjust weight based on rate and add error to total error sum
                cw.vc += nact * n.value;
                cw.vc_count++;

                err_sum += nact * cw.value;
            }

            float nv = err_sum * D_ELU(n.value);
            n.nact = nv;
            n.bc += nv; //Because the bias doesn't change, we can multiply it later.
            n.bc_count++;
        }

        /// <summary>
        /// Backpropagation similar to every other node, but skipping some lines so that it works for output nodes as well
        /// </summary>
        /// <param name="n">The node to calculate error values for</param>
        void BackPropOutputOne(Node n)
        {
            float nv = n.error * D_ELU(n.value);
            n.nact = nv;
            n.bc += nv; //Because the bias doesn't change, we can multiply it later.
            n.bc_count++;
        }

        /// <summary>
        /// Gets the error value
        /// </summary>
        /// <param name="expected">The correct output value that the network should strive towards</param>
        /// <param name="actual">The actual output value of the network</param>
        /// <returns>Returns the error value</returns>
        float GetError(float expected, float actual)
        {
            float diff = expected - actual;
            return 0.5f * diff * diff;
        }

        /// <summary>
        /// Calculates the value that a node should be(without actually editing the value of the node)
        /// </summary>
        /// <param name="n">The node to calculate the value for</param>
        /// <returns>The value of the node</returns>
        float GetNodeVal(Node n)
        {
            if (n.type == NodeType.input) return n.value;

            float sum = 0;
            foreach(Weight w in n.parents) sum += w.value * w.parent.value;
            return ELU(sum + n.bias);
        }
    }
}
