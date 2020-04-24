import time
import copy
import torch
import torch.distributed as dist

#Obtain starting time:
start_time = time.time()
#Variables used for timing nodes:
time_node_0 = 0
time_node_1 = 0
time_batch_node_0 = 0
time_batch_node_1 = 0

def main(rank, world):
######### - Master node code - #########
	if rank == 0:
		import torch
		import torchvision
		import torchvision.transforms as transforms

		transform = transforms.Compose(
		    [transforms.ToTensor(),
		     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

		trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
				                        download=True, transform=transform)
		trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
				                          shuffle=True, num_workers=2)

		testset = torchvision.datasets.CIFAR10(root='./data', train=False,
				                       download=True, transform=transform)
		testloader = torch.utils.data.DataLoader(testset, batch_size=4,
				                         shuffle=False, num_workers=2)

		classes = ('plane', 'car', 'bird', 'cat',
			   'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

		import matplotlib.pyplot as plt
		import numpy as np

		# functions to show an image


		def imshow(img):
		    img = img / 2 + 0.5     # unnormalize
		    npimg = img.numpy()
		    plt.imshow(np.transpose(npimg, (1, 2, 0)))
		    plt.show()


		# get some random training images
		dataiter = iter(trainloader)
		images, labels = dataiter.next()

		# show images
		imshow(torchvision.utils.make_grid(images))
		# print labels
		print('NODE 0: '.join('%5s' % classes[labels[j]] for j in range(4)))



		import torch.nn as nn
		import torch.nn.functional as F


		class Net(nn.Module):
			def __init__(self):
				super(Net, self).__init__()
				self.conv1 = nn.Conv2d(3, 6, 5)
				self.pool = nn.MaxPool2d(2, 2)
				self.conv2 = nn.Conv2d(6, 16, 5)
				self.fc1 = nn.Linear(16 * 5 * 5, 120)
				self.fc2 = nn.Linear(120, 84)
				self.fc3 = nn.Linear(84, 10)

			def forward(self, x):
				x = self.pool(F.relu(self.conv1(x)))
				x = self.pool(F.relu(self.conv2(x)))
				x = x.view(-1, 16 * 5 * 5)
				x = F.relu(self.fc1(x))
				x = F.relu(self.fc2(x))
				x = self.fc3(x)
				return x


		net = Net()

		import torch.optim as optim

		criterion = nn.CrossEntropyLoss()
		optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

		for epoch in range(1):  # loop over the dataset multiple times

			running_loss = 0.0
			for i, data in enumerate(trainloader, 0):
				# get the inputs; data is a list of [inputs, labels]
				inputs, labels = data

				# zero the parameter gradients
				optimizer.zero_grad()

				# forward + backward + optimize
				outputs = net(inputs)
				loss = criterion(outputs, labels)
				loss.backward()
				optimizer.step()

				# print statistics
				running_loss += loss.item()
				time_batch_node_0 = time.time() - start_time
				if i % 2000 == 1999:    # print every 2000 mini-batches
					print('NODE 0: [%d, %5d] loss: %.3f, time: %s seconds' %(epoch + 1, i + 1, running_loss / 2000,time_batch_node_0))
					running_loss = 0.0

		print('NODE 0: Finished Training')

		PATH = './cifar_net.pth'
		torch.save(net.state_dict(), PATH)

		dataiter = iter(testloader)
		images, labels = dataiter.next()

		# print images
		imshow(torchvision.utils.make_grid(images))
		print('NODE 0: GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

		net = Net()
		net.load_state_dict(torch.load(PATH))

		outputs = net(images)

		_, predicted = torch.max(outputs, 1)

		print('NODE 0: Predicted: ', ' '.join('%5s' % classes[predicted[j]]
				              for j in range(4)))

		correct = 0
		total = 0
		with torch.no_grad():
			for data in testloader:
				images, labels = data
				outputs = net(images)
				_, predicted = torch.max(outputs.data, 1)
				total += labels.size(0)
				correct += (predicted == labels).sum().item()

		print('NODE 0: Accuracy of the network on the 10000 test images: %d %%' % (
		    100 * correct / total))

		class_correct = list(0. for i in range(10))
		class_total = list(0. for i in range(10))
		with torch.no_grad():
			for data in testloader:
				images, labels = data
				outputs = net(images)
				_, predicted = torch.max(outputs, 1)
				c = (predicted == labels).squeeze()
				for i in range(4):
					label = labels[i]
					class_correct[label] += c[i].item()
					class_total[label] += 1


		for i in range(10):
		    print('NODE 0: Accuracy of %5s : %2d %%' % (
			classes[i], 100 * class_correct[i] / class_total[i]))
		time_node_0 = time.time() - start_time
		print("Total execution time for NODE 0 is: %s seconds" % time_node_0)

		#Obtain and print the weights of node 0's model:
		trained_weights_node_0 = net.fc1.weight.data
		print("Trained weights for node 0 are:")
		print(trained_weights_node_0)

		#Create a tensor to hold the weights received:
		received_weights_holder = copy.deepcopy(net.fc1.weight.data)

		#Receive and print the weights:
		dist.recv(received_weights_holder, src=1)
		print("Node 0 received the following weights from Node 1:")
		print(received_weights_holder)

		#Average the weights of the nodes:
		average_node_weights = (trained_weights_node_0 + received_weights_holder)/2
		print("Average of the node weights are:")
		print(average_node_weights)

		#Change Node 0's fc1 weight values to average of the node weights:
		model_parameters = net.state_dict()
		model_parameters['fc1.weight'].copy_(average_node_weights)
		print("Node 0's weights are now the calculated average of the node weights:")
		print(net.fc1.weight)

		#Save the changes to the model:
		net.load_state_dict(state_dict)

########### - Now train this node's model again with the new weights: - ##########

		new_weights = net.fc1.weight.data
		print("New weights are now:")
		print(new_weights)

		criterion = nn.CrossEntropyLoss()
		optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

		for epoch in range(1):  # loop over the dataset multiple times

			running_loss = 0.0
			for i, data in enumerate(trainloader, 0):
				# get the inputs; data is a list of [inputs, labels]
				inputs, labels = data

				# zero the parameter gradients
				optimizer.zero_grad()

				# forward + backward + optimize
				outputs = net(inputs)
				loss = criterion(outputs, labels)
				loss.backward()
				optimizer.step()

				# print statistics
				running_loss += loss.item()
				time_batch_node_0 = time.time() - start_time
				if i % 2000 == 1999:    # print every 2000 mini-batches
					print('NODE 0: [%d, %5d] loss: %.3f, time: %s seconds' %(epoch + 1, i + 1, running_loss / 2000,time_batch_node_0))
					running_loss = 0.0

		print('NODE 0: Finished Training')

		PATH = './cifar_net.pth'
		torch.save(net.state_dict(), PATH)

		dataiter = iter(testloader)
		images, labels = dataiter.next()

		# print images
		imshow(torchvision.utils.make_grid(images))
		print('NODE 0: GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

		net = Net()
		net.load_state_dict(torch.load(PATH))

		outputs = net(images)

		_, predicted = torch.max(outputs, 1)

		print('NODE 0: Predicted: ', ' '.join('%5s' % classes[predicted[j]]
				              for j in range(4)))

		correct = 0
		total = 0
		with torch.no_grad():
			for data in testloader:
				images, labels = data
				outputs = net(images)
				_, predicted = torch.max(outputs.data, 1)
				total += labels.size(0)
				correct += (predicted == labels).sum().item()

		print('NODE 0: Accuracy of the network on the 10000 test images: %d %%' % (
		    100 * correct / total))

		class_correct = list(0. for i in range(10))
		class_total = list(0. for i in range(10))
		with torch.no_grad():
			for data in testloader:
				images, labels = data
				outputs = net(images)
				_, predicted = torch.max(outputs, 1)
				c = (predicted == labels).squeeze()
				for i in range(4):
					label = labels[i]
					class_correct[label] += c[i].item()
					class_total[label] += 1


		for i in range(10):
		    print('NODE 0: Accuracy of %5s : %2d %%' % (
			classes[i], 100 * class_correct[i] / class_total[i]))
		time_node_0 = time.time() - start_time
		print("Total execution time for NODE 0 is: %s seconds" % time_node_0)

############## -Worker Node Code- ###############

	else:
		import torch
		import torchvision
		import torchvision.transforms as transforms

		transform = transforms.Compose(
		    [transforms.ToTensor(),
		     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

		trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
				                        download=True, transform=transform)
		trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
				                          shuffle=True, num_workers=2)

		testset = torchvision.datasets.CIFAR10(root='./data', train=False,
				                       download=True, transform=transform)
		testloader = torch.utils.data.DataLoader(testset, batch_size=4,
				                         shuffle=False, num_workers=2)

		classes = ('plane', 'car', 'bird', 'cat',
			   'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

		import matplotlib.pyplot as plt
		import numpy as np

		# functions to show an image


		def imshow(img):
		    img = img / 2 + 0.5     # unnormalize
		    npimg = img.numpy()
		    plt.imshow(np.transpose(npimg, (1, 2, 0)))
		    plt.show()


		# get some random training images
		dataiter = iter(trainloader)
		images, labels = dataiter.next()

		# show images
		imshow(torchvision.utils.make_grid(images))
		# print labels
		print('NODE 1: '.join('%5s' % classes[labels[j]] for j in range(4)))



		import torch.nn as nn
		import torch.nn.functional as F


		class Net(nn.Module):
			def __init__(self):
				super(Net, self).__init__()
				self.conv1 = nn.Conv2d(3, 6, 5)
				self.pool = nn.MaxPool2d(2, 2)
				self.conv2 = nn.Conv2d(6, 16, 5)
				self.fc1 = nn.Linear(16 * 5 * 5, 120)
				self.fc2 = nn.Linear(120, 84)
				self.fc3 = nn.Linear(84, 10)

			def forward(self, x):
				x = self.pool(F.relu(self.conv1(x)))
				x = self.pool(F.relu(self.conv2(x)))
				x = x.view(-1, 16 * 5 * 5)
				x = F.relu(self.fc1(x))
				x = F.relu(self.fc2(x))
				x = self.fc3(x)
				return x


		net = Net()

		import torch.optim as optim

		criterion = nn.CrossEntropyLoss()
		optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

		for epoch in range(1):  # loop over the dataset multiple times

			running_loss = 0.0
			for i, data in enumerate(trainloader, 0):
				# get the inputs; data is a list of [inputs, labels]
				inputs, labels = data

				# zero the parameter gradients
				optimizer.zero_grad()

				# forward + backward + optimize
				outputs = net(inputs)
				loss = criterion(outputs, labels)
				loss.backward()
				optimizer.step()

				# print statistics
				running_loss += loss.item()
				time_batch_node_1 = time.time() - start_time
				if i % 2000 == 1999:    # print every 2000 mini-batches
					print('NODE 1: [%d, %5d] loss: %.3f, time: %s seconds' %(epoch + 1, i + 1, running_loss / 2000,time_batch_node_1))
					running_loss = 0.0

		print('NODE 1: Finished Training')

		PATH = './cifar_net.pth'
		torch.save(net.state_dict(), PATH)

		dataiter = iter(testloader)
		images, labels = dataiter.next()

		# print images
		imshow(torchvision.utils.make_grid(images))
		print('NODE 1: GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

		net = Net()
		net.load_state_dict(torch.load(PATH))

		outputs = net(images)

		_, predicted = torch.max(outputs, 1)

		print('NODE 1: Predicted: ', ' '.join('%5s' % classes[predicted[j]]
				              for j in range(4)))

		correct = 0
		total = 0
		with torch.no_grad():
			for data in testloader:
				images, labels = data
				outputs = net(images)
				_, predicted = torch.max(outputs.data, 1)
				total += labels.size(0)
				correct += (predicted == labels).sum().item()

		print('NODE 1: Accuracy of the network on the 10000 test images: %d %%' % (
		    100 * correct / total))

		class_correct = list(0. for i in range(10))
		class_total = list(0. for i in range(10))
		with torch.no_grad():
			for data in testloader:
				images, labels = data
				outputs = net(images)
				_, predicted = torch.max(outputs, 1)
				c = (predicted == labels).squeeze()
				for i in range(4):
					label = labels[i]
					class_correct[label] += c[i].item()
					class_total[label] += 1


		for i in range(10):
		    print('NODE 1: Accuracy of %5s : %2d %%' % (
			classes[i], 100 * class_correct[i] / class_total[i]))
		#Get and print execution time of this node (Node 1):
		time_node_1 = time.time() - start_time
		print("Total execution time for NODE 1 is: %s seconds" % time_node_1)
		
		#Obtain and print the weigh of this node's model.
		trained_weights_node_1 = net.fc1.weight.data
		print("Trained weights for node 1 are:")
		print(trained_weights_node_1)

		#Send this node's weights to the master node:
		dist.send(trained_weights_node_1, dst=0)
		print("Node 1 has sent its model weights to Node 0.")


print("Total execution time is equal to the node that took the longest to finish.")

if __name__ == '__main__':
	dist.init_process_group(backend='mpi')
	main(dist.get_rank(), dist.get_world_size())
