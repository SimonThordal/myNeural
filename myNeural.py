import math
import random
import pprint

trainingRate = 0.1
BIAS = -1.0
pp = pprint.PrettyPrinter(indent=4)
# Very simply activation function
def activationFunction(x):
	return 1.0 / (1.0 + math.exp(-x))

def dActivationFunction(x):
	''' The derivative of the chosen activation function
	:param x: Value to invoke the function on '''
	sigma = activationFunction(x)
	return sigma*(1-sigma)

# The neuron will be able to perform summation and apply its activation function
class Neuron:
	def __init__(self):
		self.lastOutput = None
		self.lastInput = None
		self.lastOutputGradient = None
		self.lastInputGradient = None
		self.outgoingConnections = []
		self.incomingConnections = []

	def evaluate(self, inputVector):
		''' Recursively evaluate the inputs to the neurons, so that the first input layer gets evaluated first
		and passes its output times the weight of the connection up through the graph
		:param inputVector: A vector of input values used in the bottom layer'''
		_input = 0.0
		for connection in self.incomingConnections:
			_input += connection.source.evaluate(inputVector)*connection.weight
		self.lastInput = _input
		self.lastOutput = activationFunction(self.lastInput)
		return self.lastOutput

	def calcOutputGradient(self):
		''' Compute the partial derivative of the loss of the output of the neuron
		:param target: the true target class of the input'''
		self.lastOutputGradient = 0.0
		for connection in self.outgoingConnections:
			self.lastOutputGradient += connection.target.lastInputGradient*connection.lastWeight


	def calcInputGradient(self):
		''' Compute the partial derivative of the loss of the input to the neuron
		:param target: the true target class of the input'''
		self.lastInputGradient = self.lastOutputGradient*dActivationFunction(self.lastInput)

	def backpropagate(self):
		self.calcOutputGradient()
		self.calcInputGradient()
		[connection.backpropagate() for connection in self.incomingConnections]


	def __str__(self):
		t = "output: {0:.4f}".format(self.lastOutput)
		for i in range(0, len(self.incomingConnections)):
			t += " weight {0}: {1:.4f}".format(i, self.incomingConnections[i].weight)
		t += "\n"
		return t


# A neuron which always evaluates to 1.0 to conserve the weight stored in the bias connection
class BiasNeuron(Neuron):
	def __init__(self):
		Neuron.__init__(self)
		self.lastOutput = 1.0

	def evaluate(self, inputVector):
		return self.lastOutput
	
	def backpropagate(self):
		''' Nowhere to propagate too'''
		pass

# An input neuron with no incoming connections
class InputNeuron(Neuron):
	def __init__(self, index):
		Neuron.__init__(self)
		self.index = index

	def evaluate(self, inputVector):
		self.lastOutput = inputVector[self.index]
		return self.lastOutput


# An output neuron
class OutputNeuron(Neuron):
	def __init__(self, index):
		Neuron.__init__(self)
		self.index = index

	def calcOutputGradient(self, target):
		''' Compute the partial derivative of the loss of the output of the neuron
		:param target: the true target class of the input'''
		if target != self.index:
			self.lastOutputGradient = 0.0
		else:
			self.lastOutputGradient = -1.0/self.lastOutput
		return self.lastOutputGradient

	def calcInputGradient(self, target):
		''' Compute the partial derivative of the loss of the input to the neuron
		:param target: the true target class of the input'''
		if target != self.index:
			self.lastInputGradient = -1*(-self.lastOutput)
		else:
			self.lastInputGradient = -1*(1-self.lastOutput)
		return self.lastInputGradient

	def backpropagate(self, target):
		self.calcOutputGradient(target)
		self.calcInputGradient(target)
		[connection.backpropagate() for connection in self.incomingConnections]



# The layer will basically only act as a container for neurons
class Layer:
	def __init__(self, nrOfNeurons):
		self.neurons = [Neuron() for _ in range(0, nrOfNeurons)]
		self.neurons.insert(0, BiasNeuron())
		self.prevLayer = None
		self.nextLayer = None

	def __str__(self):
		t = ""
		for i in range(0, len(self.neurons)):
			n = self.neurons[i]
			t += "Neuron {0}: {1}".format(i+1, str(n))
		return t

	def backpropagate(self):
		[neuron.backpropagate() for neuron in self.neurons]
		self.prevLayer.backpropagate()

	def connect(self):
		for neuron in self.neurons:
			for outNeuron in self.nextLayer.neurons:
				if not isinstance(outNeuron, BiasNeuron):
					Connection(neuron, outNeuron)

# The input layer which does not have inputs of its own
class InputLayer(Layer):
	def __init__(self, nrOfInputs):
		self.neurons = [InputNeuron(i) for i in range(0, nrOfInputs)]
		self.neurons.insert(0, BiasNeuron())

	def backpropagate(self):
		''' Nowhere to propagate too'''
		pass




class OutputLayer(Layer):
	def __init__(self, nrOfOutputs):
		self.neurons = [OutputNeuron(i) for i in range(0, nrOfOutputs)]

	def backpropagate(self, target):
		[neuron.backpropagate(target) for neuron in self.neurons]
		self.prevLayer.backpropagate()

	def connect(self):
		pass

# The net handles all the higher level stuff
class NeuralNet:
	def __init__(self, nrOfInputs, nrOfLayers, neuronsPerLayer, nrOfOutputs):
		self.nrOfInputs = nrOfInputs
		self.nrOfLayers = nrOfLayers
		self.neuronsPerLayer = neuronsPerLayer

		# Create the layers in the neural net
		self.layers = []
		self.layers.append(InputLayer(nrOfInputs))
		for i in range(0, nrOfLayers):
			self.layers.append(Layer(neuronsPerLayer))
		self.layers.append(OutputLayer(nrOfOutputs))

		# Connect the layer super graph
		self.layers[0].nextLayer = self.layers[1]
		for i in range(1, nrOfLayers+1):
			self.layers[i].prevLayer = self.layers[i-1]
			self.layers[i].nextLayer = self.layers[i+1]
		self.layers[nrOfLayers+1].prevLayer = self.layers[nrOfLayers]

		# Create connections between the nodes in the layers
		for layer in self.layers:
			layer.connect()
		

	def evaluate(self, data):
		self.target = data['outputTarget']
		inputVector = data['inputs']
		return map(lambda o: o.evaluate(inputVector), self.layers[self.nrOfLayers+1].neurons)

	def train(self):
		self.layers[self.nrOfLayers+1].backpropagate(self.target)
		#[neuron.backpropagate(self.target) for neuron in self.layers[self.nrOfLayers+1].neurons]

	def __str__(self):
		t = "Network structure:\n"
		for i in range(0, self.nrOfLayers + 2):
			t += "Layer {0}:\n".format(i) + str(self.layers[i]) + "\n"
		return t

class Connection:
	def __init__(self, source, target):
		self.lastWeight = None

		if isinstance(source, BiasNeuron):
			self.weight = BIAS
		else:
			self.weight = random.uniform(0,1)

		self.source = source
		self.target = target

		self.source.outgoingConnections.append(self)
		self.target.incomingConnections.append(self)

	def computeGradient(self):
		if isinstance(self.source, BiasNeuron):
			self.lastWeight = self.weight
			self.weight -= trainingRate*self.target.lastInputGradient
		else:
			self.lastWeight = self.weight
			self.weight -= trainingRate*self.target.lastInputGradient*self.source.lastOutput

	def setWeight(self, weight):
		self.weight = weight

	def backpropagate(self):
		self.computeGradient()
		#self.source.backpropagate()

def createFiftyFiftyTestCase(nrInputs):
	coin = random.uniform(0,1)
	inp = None
	outp = None
	if coin <= 0.5:
		inp = [random.uniform(0.0, 0.5) for _ in range(0, nrInputs)]
		outp = 0
	else:
		inp = [random.uniform(0.50001, 1.0) for _ in range(0, nrInputs)]
		outp = 1
	return {'inputs':inp, 'outputTarget':outp}

def createAndTestCase():
	inputs = [random.randint(0,1) for _ in range(0,2)]
	if inputs[0] == inputs[1]:
		outp = 1
	else:
		outp = 0
	return {'inputs':inputs, 'outputTarget': outp}

def createXORTestCase():
	inputs = [random.randint(0,1) for _ in range(0,2)]
	if inputs[0] != inputs[1]:
		outp = 1
	else:
		outp = 0
	return {'inputs':inputs, 'outputTarget': outp}

def trainAndNetwork(it):
	net = NeuralNet(2,1,2,2)
	for _ in range(0, it):
		data = createAndTestCase()
		outputs = net.evaluate(data)
		print "I1: {}, I2: {}, Target: {}, P0: {:.4f}, P1: {:.4f}".format(
																		data['inputs'][0],
																		data['inputs'][1],
																		data['outputTarget'], 
																		outputs[0], 
																		outputs[1])
		net.train()
	print
	print net

def trainXORNetwork(it):
	net = NeuralNet(2,1,2,2)
	for i in range(0, it):
		data = createXORTestCase()
		outputs = net.evaluate(data)
		if i % 1000 == 0:
			print "I1: {}, I2: {}, Target: {}, P0: {:.4f}, P1: {:.4f}".format(
																		data['inputs'][0],
																		data['inputs'][1],
																		data['outputTarget'], 
																		outputs[0], 
																		outputs[1])
		net.train()
	print
	print net

def trainFiftyFiftyNetwork(it):
	net = NeuralNet(1,1,2,2)
	for _ in range(0, it):
		data = createFiftyFiftyTestCase(1)
		outputs = net.evaluate(data)
		print "Input: {:.4f}, Target: {}, P0: {:.4f}, P1: {:.4f}".format(data['inputs'][0], data['outputTarget'], outputs[0], outputs[1])
		net.train()
	print
	print net




