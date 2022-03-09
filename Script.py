def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, scheduler=None, early_stopping=None):
    since = time.time()
    
    training_accuracies, training_losses, val_accuracies, val_losses = [], [], [], []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 60)
        
        
        for phase in ['Training', 'Validation']:
            if phase == 'Training':
                model.train()  
                dataloaders = train_loader
                dataset_sizes = len(training_dataset)
            else:
                model.eval()   #
                dataloaders = val_loader
                dataset_sizes = len(validation_dataset)

            running_loss = 0.0
            running_corrects = 0
     
            
            for inputs, labels in tqdm(dataloaders):
                inputs = inputs.to(device)
                labels = labels.to(device)

                
                optimizer.zero_grad()

                
                with torch.set_grad_enabled(phase == 'Training'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    
                    if phase == 'Training':
                        loss.backward()
                        optimizer.step()

                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'Training':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes
            epoch_acc = running_corrects.double() / dataset_sizes

            if phase == 'Training':
                training_accuracies.append(epoch_acc)
                training_losses.append(epoch_loss)
            else:
                val_accuracies.append(epoch_acc)
                val_losses.append(epoch_loss)

            
            if phase == 'Validation' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

            
            print('{} Loss after epoch: {:.4f}, Acc after epoch: {:.4f}\n'.format(
            phase, epoch_loss, epoch_acc))
            
        
        
        if early_stopping is not None:
            early_stopping(val_losses[-1])

            if early_stopping.early_stop:
                print('Early Stopping Initiated')
                break


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    
    
    
    model.load_state_dict(best_model_wts)
    return model, training_accuracies, training_losses, val_accuracies, val_losses
    
def main():

	resnet18 = torchvision.models.resnet18(pretrained=False)

	examples = torch.rand(1,3,224,224)

	resnet18_traced = torch.jit.trace(resnet18, example_inputs = examples)

	resnet18_traced.save("resnet18.traced.pt")

	train_path = 'train/'
	test_path = 'test/'

	classes = os.listdir(train_path)
	print(classes)

	num_classes = len(classes)


	print("_______________________")
	print("Training Images")
	print("_______________________")

	for animal in os.listdir(train_path):
    	print(f'Number of {animal} images: {len(os.listdir(train_path + "/" + animal))}')


	print("_______________________")
	print("Test Images")
	print("_______________________")
	# See the number of images for each class
	for animal in os.listdir(test_path):
    	print(f'Number of {animal} images: {len(os.listdir(test_path + "/" + animal))}')



	# Split size for training, validation, test
	split_size = 0.08

	train_data, val_data= data_split(train_path, split_size=split_size)
 

	# To convert string labels into integers
	le = preprocessing.LabelEncoder()
	le.fit(classes)


	# Train, Validation, test Dataset
	training_dataset = _Dataset(train_data,train_path, transform=data_transforms['train'])
	validation_dataset = _Dataset(val_data,train_path, transform=data_transforms['val'])
	test_dataset = torchvision.datasets.ImageFolder(root=test_path,transform=test_transforms)


	print("_______________________")
	print("_______________________")
	print("_______________________")
	print('Number of training images: {}'.format(len(train_data)))
	print('Number of validation images: {}'.format(len(val_data)))
	print('Number of Test images: {}'.format(len(test_dataset)))


if __name__ == "__main__":
	main()