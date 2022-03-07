import torch
import torchvision
import torch.nn.functional as f
from sklearn.model_selection import train_test_split

class cnn(nn.Module):
	def startModel():
		super(cnn, self).__init__()
		self.conv_layer1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size = 3)
		self.conv_layer2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size= 3)
		self.max_pool1 = nn.MaxPool2d(kernal_size = 2, stride = 2)
		
		self.conv_layer3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size = 3)
		self.conv_layer4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size = 3)
		self.max_pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)
		
		self.fc1 = nn.Linear(1600, 128)
		self.relu1 = nn.ReLU()
		self.fc2 = nn.Linear(128, num_classes)
		
		resnet18 = torchvision.models.resnet18(pretrained=True)

		examples = torch.rand(1,3,224,224)

		resnet18_traced = torch.jit.trace(resnet18, example_inputs = examples)

		resnet18_traced.save("resnet18.traced.pt")
		
		#May go elsewhere
		model = ConvNeuralNet(num_classes)
		
		criterion = nn.CrossEntropyLoss()
		
		optimizer = torch.optim.SGD(model.parameters(), lr= learning_rate, weight_decay = 0.005, momentum = 0.9)
		total_step = len(train_loader)
	
	def fwd(self, x):
		out = self.conv_layer1(x)
		out = self.conv_layer2(out)
		out = self.max_pool1(out)
		
		out = self.conv_layer3(out)
		out = self.conv_layer4(out)
		out = self.max_pool2(out)
		
		out = out.reshape(out.size(0), -1)
		
		out = self.fc1(out)
		out = self.relu1(out)
		out = self.self.fc2(out)
		return out
		
	def train(images, labels, num_epochs, train_loader):
		for epoch in range(num_epochs):
			for i, (images, labels) in enumerate(train_loader):
				images = images.to(device)
				labels = labels.to(device)
				
				outputs = model(images)
				loss = criterion(outputs, labels)
				
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
				
	def train(trainloader, images, labels, model):
		affirmativeMatch = 0
		total = 0
		for images, labels in trainloader:
			images = images.to(device)
			labels = labels.to(device)
			outputs = model(images)
			_, predicted = torch.max(outputs.data, 1)
			total += labels.size(0)
			correct += (predicted == labels).sum().item()
			
if __name__ == "__main__":
#dataset path
	path = r'~/Desktop/Andrei/Dataset/'
	dataset = scilearn.imread(path)
	
	training, testing = train_test_split(df, test_size = 0.2, random_state = 25)
	images = []
	labels = []
	numtraining = training.shape[0]
	numtesting = testing.shape[0]
	print(numtraining)
	print(numtesting)
	train(images, labels, 2, dataset)
	