using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace LyNN
{
    public enum NodeType
    {
        input,
        node,
        output,
    }

    /// <summary>
    /// Delegate activation function to create custom functions(note that input nodes do not use this function)
    /// </summary>
    /// <param name="val">The value given to this activation function</param>
    /// <param name="self">The node that's calling this function. Mostly here for the possibility of doing wacky stuff like memory cells or something</param>
    /// <returns></returns>
    public delegate double actfunc(double val, Node self);

    public class Node
    {
        public NodeType type;
        public double value;
        public List<Weight> parents;
        public List<Weight> children;
        public double bias;
        public actfunc af;
        public actfunc d_af;
        public ActivationFunction actfunc;

        //error values
        public double error;
        public double nact;
        public double bc;
        public int bc_count;

        //Extra custom values for some possibly wacky nodes
        public List<double> customval = new List<double>();
    }

    public class Weight
    {
        public double value;
        public Node parent;
        public Node child;

        public double vc;
        public int vc_count;
    }

    public enum ActivationFunction
    {
        Sigmoid = 1,
        ELU = 101,
        ReLU = 102,

        Custom1 = -1,
        Custom2 = -2,
        Custom3 = -3,
        //Etc... Going on the assumption here that nobody would *ever* need that many custom activation functions in one network
    }

    public class Network
    {
        public int numInputs;
        public int numOutputs;
        public int numHiddenLayers;

        private List<Node> inputs;
        private List<Node> outputs;
        private List<List<Node>> allNodes;
        private Random rand = new Random();

        private List<actfunc> CustomFunctions_d = new List<actfunc>();
        private List<actfunc> CustomFunctions = new List<actfunc>();

        //Clipping values to try prevention of gradient explosion
        public double gradient_clipping = 0.2;
        public double weight_clipping = 1.0;
        public double activation_clipping = 2.0;

        /// <summary>
        /// Builds a network with the given amount of inputs, outputs, and hidden layers
        /// </summary>
        /// <param name="numInputs">Amount of inputs the network should have</param>
        /// <param name="layers">An array containing the amount of hidden nodes per layer</param>
        /// <param name="numOutputs">Amount of outputs the network should have</param>
        /// <param name="actfunc">The activation function to use</param>
        /// <returns>Returns a Network object</returns>
        public static Network BuildNetwork(int numInputs, int[] layers, int numOutputs, ActivationFunction actfunc = ActivationFunction.ELU)
        {
            Network net = new Network();
            net.Build(numInputs, layers, numOutputs, actfunc);
            return net;
        }

        public void Build(int numInputs, int[] layers, int numOutputs, ActivationFunction actfunc = ActivationFunction.ELU)
        {
            //Sanity checking arguments
            if (numInputs < 1) throw new Exception("Can't have less than one input");
            if (numOutputs < 1) throw new Exception("Can't have less than one output");

            //Create and initialize the network
            this.numInputs = numInputs;
            this.numOutputs = numOutputs;
            this.numHiddenLayers = layers.Length;
            this.inputs = new List<Node>();
            this.outputs = new List<Node>();
            this.allNodes = new List<List<Node>>();

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

                    n.af = this.ETF(actfunc, false);
                    n.d_af = this.ETF(actfunc, true);
                    n.actfunc = actfunc;

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
                    if (t == NodeType.input) this.inputs.Add(n);
                    if (t == NodeType.output) this.outputs.Add(n);

                    //Add to 'news' list just so that it can be used as a reference for the nodes in the next layer
                    news.Add(n);
                }

                //Add the new layer of nodes to the allNodes list
                this.allNodes.Add(news);

                prevs.Clear();
                prevs.AddRange(news);
            }
        }

        /// <summary>
        /// Randomizes the weights and biases for the network
        /// </summary>
        /// <param name="mulbias">The amount to multiply the randomized biases with(if 1, the values will be between -0.5 and 0.5)</param>
        /// <param name="mulweight">The amount to multiply the randomized weights with(if 1, the values will be between -0.5 and 0.5)</param>
        public void RandomizeNetwork(double mulbias = 1, double mulweight = 1)
        {
            for(int i = 0; i < allNodes.Count; i++)
            {
                for (int j = 0; j < allNodes[i].Count; j++)
                {
                    //Iterate through all nodes and randomize their bias
                    Node n = (allNodes[i])[j];
                    n.bias = (rand.NextDouble() - 0.5) * mulbias;

                    //Then iterate through all of its children weights and randomize them
                    for (int k = 0; k < n.children.Count; k++)
                    {
                        Weight w = n.children[k];
                        w.value = (rand.NextDouble() - 0.5) * mulweight;
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
            StringBuilder nt = new StringBuilder();
            for(int i = 0; i < allNodes.Count; i++)
            {
                List<Node> nn = allNodes[i];

                //Write the amount of nodes in this layer on one line
                nt.Append(nn.Count + "\n");
                foreach(Node n in nn)
                {
                    //Then, for each node, write the bias, value function ID, and possibly its custom variable values on one line
                    nt.Append(n.bias + ";" + (int)n.actfunc);
                    for(int j = 0; j < n.customval.Count; j++) nt.Append(";" + n.customval[j]);
                    nt.Append("\n");

                    //Write all of its children weights in order of the node's index
                    if (n.children.Count == 0)
                    {
                        nt.Append(";");
                    }
                    else
                    {
                        foreach (Weight w in n.children)
                        {
                            nt.Append(w.value.ToString() + ";");
                        }
                    }
                    nt.Append("\n");
                }
                if(i < allNodes.Count - 1) nt.Append("\n");
            }
            File.WriteAllText(name, nt.ToString());
        }

        /// <summary>
        /// Loads a network from a file
        /// </summary>
        /// <param name="name">The filename to read from</param>
        /// <param name="def_actfunc">The value function to fall back to in case a node doesn't have one assigned</param>
        public static Network LoadNetwork(string name, ActivationFunction def_actfunc = ActivationFunction.ELU)
        {
            Network nw = new Network();
            nw.Load(name, def_actfunc);
            return nw;
        }

        public void Load(string name, ActivationFunction def_actfunc = ActivationFunction.ELU)
        {
            //Create and initialize the network
            this.inputs = new List<Node>();
            this.outputs = new List<Node>();
            this.allNodes = new List<List<Node>>();

            //Read file
            //The layers are split by a double newline
            string[] all = File.ReadAllText(name).Split(new string[] { "\n\n" }, StringSplitOptions.RemoveEmptyEntries);
            this.numHiddenLayers = all.Length - 2;

            for(int i = 0; i < all.Length; i++)
            {
                this.allNodes.Add(new List<Node>());
            }

            for(int layer = all.Length - 1; layer >= 0; layer--)
            {
                string[] lines = all[layer].Split(new char[] { '\n' }, StringSplitOptions.RemoveEmptyEntries);

                int num = int.Parse(lines[0]);

                //Add to numinputs/outputs vars
                if (layer == 0) this.numInputs = num;
                if (layer == all.Length - 1) this.numOutputs = num;

                for (int i = 0; i < num; i++)
                {
                    Node n = new Node();
                    n.children = new List<Weight>();
                    n.parents = new List<Weight>();

                    if (layer == 0)
                    {
                        //Input nodes get marked as input nodes and added to the list of input nodes
                        this.inputs.Add(n);
                        n.type = NodeType.input;
                    }
                    else if (layer == all.Length - 1)
                    {
                        //Output nodes get marked as output nodes and added to the list of output nodes
                        this.outputs.Add(n);
                        n.type = NodeType.output;
                    }
                    else n.type = NodeType.node; //Everything else is a regular node

                    //Add this node to the list of all nodes, then start reading the values out of the file
                    this.allNodes[layer].Add(n);

                    //Backwards compatibility check
                    string biasline = lines[i * 2 + 1];
                    if (biasline.Contains(";"))
                    {
                        //Set bias and value function
                        string[] biassplit = biasline.Split(new char[] { ';' });
                        n.bias = double.Parse(biassplit[0]);
                        n.actfunc = (ActivationFunction)int.Parse(biassplit[1]);

                        //If there's any more on that line, that means there's custom values.
                        for(int j = 2; j < biassplit.Length; j++) n.customval.Add(double.Parse(biassplit[j]));
                    }
                    else
                    {
                        //Fall back to the default value function
                        n.bias = double.Parse(biasline);
                        n.actfunc = def_actfunc;
                    }

                    //Assign value function(and complementary derivative) to the node
                    n.af = this.ETF(n.actfunc, false);
                    n.d_af = this.ETF(n.actfunc, true);


                    if (i * 2 + 2 < lines.Length)
                    {
                        //Add all of the children weights in order
                        string[] ws = lines[i * 2 + 2].Split(new char[] { ';' }, StringSplitOptions.RemoveEmptyEntries);
                        for(int j = 0; j < ws.Length; j++)
                        {
                            Weight w = new Weight();
                            w.parent = n;
                            w.child = this.allNodes[layer + 1][j];
                            w.value = double.Parse(ws[j]);
                            n.children.Add(w);
                            w.child.parents.Add(w);
                        }
                    }
                }
            }
        }

        


        /// <summary>
        /// Adds a custom activation function and returns the ID it has been given
        /// </summary>
        /// <param name="Function"></param>
        /// <param name="Derivative"></param>
        public ActivationFunction AddCustomActivation(actfunc Function, actfunc Derivative)
        {
            CustomFunctions.Add(Function);
            CustomFunctions_d.Add(Derivative);
            return (ActivationFunction)(-CustomFunctions.Count);
        }


        /// <summary>
        /// Gets the function attached to the given activation function value
        /// </summary>
        /// <param name="actfunc">The activation function enum value</param>
        /// <param name="d">Whether or not this should return the derivative of the function</param>
        /// <returns></returns>
        private actfunc ETF(ActivationFunction actfunc, bool d)
        {
            //Custom activation functions
            if(actfunc < 0)
            {
                int id = -1 - (int)actfunc;
                if (id > CustomFunctions.Count) throw new IndexOutOfRangeException("Custom function ID doesn't exist!");

                if (d) return CustomFunctions_d[id];
                return CustomFunctions[id];
            }

            //Pre-programmed activation functions
            switch(actfunc)
            {
                case ActivationFunction.ELU:
                    if (d) return D_ELU;
                    return ELU;

                case ActivationFunction.ReLU:
                    if (d) return D_ReLU;
                    return ReLU;

                default:
                case ActivationFunction.Sigmoid:
                    if (d) return D_Sigmoid_e;
                    return Sigmoid_e;
            }
        }



        /// <summary>
        /// Exponential-based sigmoid activation function
        /// </summary>
        public double Sigmoid_e(double val, Node self) { return (1 / (1 + Math.Exp(-val))); }

        /// <summary>
        /// ELU rectifier activation function
        /// </summary>
        public double ELU(double val, Node self)
        {
            if (val > activation_clipping) return activation_clipping;
            if (val >= 0) return val;
            else return (Math.Exp(val) - 1);
        }

        /// <summary>
        /// ReLU rectifier activation function
        /// </summary>
        public double ReLU(double val, Node self)
        {
            if (val > activation_clipping) return activation_clipping;
            if (val >= 0) return val;
            else return 0;
        }


        /// <summary>
        /// Derivative of the exponential-based sigmoid activation function
        /// </summary>
        public double D_Sigmoid_e(double val, Node self) { return val * (1 - val); }

        /// <summary>
        /// Derivative of the ELU rectifier function
        /// </summary>
        public double D_ELU(double val, Node self)
        {
            if (val > activation_clipping) return 0;
            if (val >= 0) return 1;
            else return Math.Exp(val);
        }

        /// <summary>
        /// Derivative of the ReLU rectifier function
        /// </summary>
        public double D_ReLU(double val, Node self)
        {
            if (val > activation_clipping) return 0;
            if (val >= 0) return 1;
            else return 0;
        }

        /// <summary>
        /// Evaluates what the network thinks about the inputs
        /// </summary>
        /// <param name="inputs">The inputs to evaluate</param>
        /// <returns>Returns the output of the network</returns>
        public double[] Evaluate(double[] inputs)
        {
            if (inputs.Length != numInputs) throw new Exception("Incorrect number of inputs given! Got " + inputs.Length.ToString() + ", expected " + numInputs.ToString());

            //Set input node values
            for (int i = 0; i < inputs.Length; i++) this.inputs[i].value = inputs[i];

            //Calculate node value for each node in order(skip input nodes because these obviously do not get calculated)
            for (int i = 1; i < allNodes.Count; i++)
            {
                List<Node> ani = allNodes[i];
                foreach(Node n in ani)
                {
                    //Sum all weights multiplied by their parent node, update the node value so that the next nodes can use this value(otherwise known as forward propagation)
                    double sum = 0;
                    foreach (Weight w in n.parents) sum += w.value * w.parent.value;
                    n.value = n.af(sum + n.bias, n);
                }
            }

            //Read the output nodes and write them into a double array to return
            double[] ret = new double[numOutputs];
            for (int i = 0; i < outputs.Count; i++) ret[i] = outputs[i].value;

            return ret;
        }

        /// <summary>
        /// Takes in training data and sets the error values accordingly
        /// </summary>
        /// <param name="inputs">Training data inputs</param>
        /// <param name="goodOutputs">Training data outputs(that the network should strive to get right)</param>
        /// <returns>Returns the total error in the output, lower is better</returns>
        public double TrainNetwork(double[] inputs, double[] goodOutputs)
        {
            //Feed the inputs to the network and evaluate
            double[] rets = Evaluate(inputs);

            //Calculate the total error to return later
            double errorsum = 0;
            for (int i = 0; i < numOutputs; i++)
            {
                double diff = goodOutputs[i] - rets[i];
                outputs[i].error = diff;
                errorsum += diff * diff;
            }

            //Go through all output nodes, adjust the biases and set nact values.
            for (int j = 0; j < numOutputs; j++)
            {
                BackPropOutputOne(outputs[j]);
            }

            //Go through all nodes backwards(from output to input, but skipping the output nodes) and adjust the weights based on the error values
            for (int i = numHiddenLayers; i >= 0; i--)
            {
                List<Node> nodes = allNodes[i];
                for (int j = 0; j < nodes.Count; j++)
                {
                    BackPropOne(nodes[j]);
                }
            }

            return 0.5f * errorsum;
        }

        /// <summary>
        /// Applies the average of all changes that the last training set proposed
        /// </summary>
        /// <param name="rate">The rate at which to change</param>
        public void ApplyTrainingChanges(double rate)
        {
            for (int i = numHiddenLayers + 1; i >= 0; i--)
            {
                List<Node> nodes = allNodes[i];
                for (int j = 0; j < nodes.Count; j++)
                {
                    Node n = nodes[j];
                    for (int x = 0; x < n.children.Count; x++)
                    {
                        Weight cw = n.children[x];
                        double cwv = rate * (cw.vc / cw.vc_count);

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
                    double bcv = rate * (n.bc / n.bc_count);

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
            double err_sum = 0;
            double nval = n.value;
            for (int i = 0; i < n.children.Count; i++)
            {
                Weight cw = n.children[i];
                double nact = cw.child.nact;

                //Adjust weight based on rate and add error to total error sum
                cw.vc += nact * nval;
                cw.vc_count++;

                err_sum += nact * cw.value;
            }

            double nv = err_sum * n.d_af(nval, n);
            n.nact = nv;
            n.bc += nv;
            n.bc_count++;
        }

        /// <summary>
        /// Backpropagation similar to every other node, but skipping some lines so that it works for output nodes as well
        /// </summary>
        /// <param name="n">The node to calculate error values for</param>
        void BackPropOutputOne(Node n)
        {
            double nv = n.error * n.d_af(n.value, n);
            n.nact = nv;
            n.bc += nv;
            n.bc_count++;
        } 
    }
}
